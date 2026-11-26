"""bioRxiv / medRxiv full text from the official text-mining S3 buckets.

bioRxiv's site blocks automated access; its sanctioned machine route is a
pair of requester-pays buckets (billed to YOUR AWS account):

    s3://biorxiv-src-monthly/Current_Content/<Month_Year>/<uuid>.meca
    s3://medrxiv-src-monthly/Current_Content/<Month_Year>/<uuid>.meca

Each ``.meca`` is a zip (MECA package): manifest, JATS XML, PDF, figures.
There is no DOI -> package index, so one is built per monthly folder by
reading only a few KB of each package with ranged GETs (the zip's central
directory, then the head of the JATS entry, where the DOI sits) and cached
in ``<LC_CACHE_DIR>/biorxiv_s3/``. Indexing a month (~4-5k packages) moves
roughly 150 MB: about $0.01-0.02 of egress plus ~$0.004 of requests. Only
the matched paper's JATS is downloaded in full.

Opt-in with ``LC_BIORXIV_S3=1``; credentials come from the standard AWS
chain (``aws configure``, ``AWS_PROFILE``, or ``AWS_ACCESS_KEY_ID``/...).
"""

from __future__ import annotations

import calendar
import json
import logging
import os
import re
import struct
import zlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

BUCKETS = ("biorxiv-src-monthly", "medrxiv-src-monthly")
REGION = "us-east-1"
_RP = {"RequestPayer": "requester"}
_DOI_IN_JATS = re.compile(rb'<article-id[^>]*pub-id-type="doi"[^>]*>\s*([^<\s]+)\s*<')
_JATS_HEAD_BYTES = 24_000  # compressed bytes; enough to reach <article-meta>

Reader = Callable[[int, int], bytes]  # (start, end_inclusive) -> bytes


# ---------------------------------------------------------------------------
# Remote zip: read entries from a zip given only ranged reads
# ---------------------------------------------------------------------------


def zip_entries(read: Reader, size: int) -> dict[str, tuple[int, int, int, int]]:
    """name -> (local_header_offset, compressed_size, uncompressed_size, method)."""
    tail_start = max(0, size - 66_000)
    tail = read(tail_start, size - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise ValueError("not a zip (no end-of-central-directory record)")
    cd_size, cd_offset = struct.unpack_from("<II", tail, eocd + 12)
    if cd_offset == 0xFFFFFFFF:
        raise ValueError("zip64 packages are not supported")
    if cd_offset >= tail_start:
        cd = tail[cd_offset - tail_start : cd_offset - tail_start + cd_size]
    else:
        cd = read(cd_offset, cd_offset + cd_size - 1)
    entries, pos = {}, 0
    while pos + 46 <= len(cd) and cd[pos : pos + 4] == b"PK\x01\x02":
        (
            _,
            _,
            _,
            _,
            method,
            _,
            _,
            _,
            csize,
            usize,
            nlen,
            elen,
            clen,
            _,
            _,
            _,
            offset,
        ) = struct.unpack_from("<4s6H3I5HII", cd, pos)
        name = cd[pos + 46 : pos + 46 + nlen].decode("utf-8", "replace")
        entries[name] = (offset, csize, usize, method)
        pos += 46 + nlen + elen + clen
    return entries


def read_entry(
    read: Reader, entry: tuple[int, int, int, int], limit: int | None = None
) -> bytes:
    """Decompressed bytes of one entry (only the first ``limit`` compressed
    bytes when given: a partial but valid prefix of the file)."""
    offset, csize, _, method = entry
    header = read(offset, offset + 29)
    nlen, elen = struct.unpack_from("<HH", header, 26)
    start = offset + 30 + nlen + elen
    n = csize if limit is None else min(csize, limit)
    if n == 0:
        return b""
    data = read(start, start + n - 1)
    if method == 0:
        return data
    if method == 8:
        return zlib.decompressobj(-15).decompress(data)
    raise ValueError(f"unsupported zip compression method {method}")


def jats_entry(entries: dict) -> tuple[str, tuple] | None:
    """The package's JATS article: the largest .xml under content/."""
    xmls = [
        (n, e)
        for n, e in entries.items()
        if n.lower().endswith(".xml")
        and "content/" in n.lower()
        and "manifest" not in n.lower()
    ]
    return max(xmls, key=lambda x: x[1][2]) if xmls else None


# ---------------------------------------------------------------------------
# S3 access + per-month DOI index
# ---------------------------------------------------------------------------


def _s3():
    import boto3  # lazy: optional dependency, only for this opt-in source

    return boto3.client("s3", region_name=REGION)


def _reader(s3, bucket: str, key: str) -> Reader:
    def read(start: int, end: int) -> bytes:
        obj = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end}", **_RP)
        return obj["Body"].read()

    return read


def _package_doi(s3, bucket: str, key: str, size: int) -> str:
    try:
        read = _reader(s3, bucket, key)
        found = jats_entry(zip_entries(read, size))
        if not found:
            return ""
        head = read_entry(read, found[1], limit=_JATS_HEAD_BYTES)
        m = _DOI_IN_JATS.search(head)
        return m.group(1).decode().lower() if m else ""
    except Exception as e:
        logger.debug("biorxiv_s3: could not index %s: %s", key, e)
        return ""


def _month_prefixes(s3, bucket: str, doi: str) -> list[str]:
    """Folders for the DOI's posting month and the next (late-month postings)."""
    m = re.search(r"/(\d{4})\.(\d{2})\.\d{2}\.", doi)
    if not m:
        return []
    y, mo = int(m.group(1)), int(m.group(2))
    wanted = [(y, mo), (y + (mo == 12), mo % 12 + 1)]
    names = {f"{calendar.month_name[mm]}_{yy}".lower() for yy, mm in wanted}
    resp = s3.list_objects_v2(
        Bucket=bucket, Prefix="Current_Content/", Delimiter="/", **_RP
    )
    prefixes = [p["Prefix"] for p in resp.get("CommonPrefixes", [])]
    return [p for p in prefixes if p.rstrip("/").rsplit("/", 1)[-1].lower() in names]


def _index_path(bucket: str, prefix: str) -> Path:
    base = Path(os.getenv("LC_CACHE_DIR", ".lc_cache")) / "biorxiv_s3"
    return base / f"{bucket}__{prefix.strip('/').replace('/', '__')}.json"


def index_folder(s3, bucket: str, prefix: str) -> dict[str, str]:
    """{package key: doi} for one monthly folder; incremental (new keys only)."""
    path = _index_path(bucket, prefix)
    try:
        index = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        index = {}
    todo = []
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix, **_RP
    ):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".meca") and obj["Key"] not in index:
                todo.append((obj["Key"], obj["Size"]))
    if todo:
        logger.warning(
            "biorxiv_s3: indexing %d new packages in s3://%s/%s (requester pays)",
            len(todo),
            bucket,
            prefix,
        )
        with ThreadPoolExecutor(max_workers=16) as pool:
            dois = pool.map(lambda kv: _package_doi(s3, bucket, *kv), todo)
            for (key, _), doi in zip(todo, dois, strict=True):
                index[key] = doi
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(index))
    return index


def fetch_jats(doi: str) -> str | None:
    """Full JATS XML for a bioRxiv/medRxiv DOI, or None (not found / no access)."""
    doi = doi.lower()
    try:
        s3 = _s3()
        for bucket in BUCKETS:
            for prefix in _month_prefixes(s3, bucket, doi):
                for key, found_doi in index_folder(s3, bucket, prefix).items():
                    if found_doi == doi:
                        read = _reader(s3, bucket, key)
                        size = s3.head_object(Bucket=bucket, Key=key, **_RP)[
                            "ContentLength"
                        ]
                        entry = jats_entry(zip_entries(read, size))
                        if entry:
                            return read_entry(read, entry[1]).decode("utf-8", "replace")
    except Exception as e:
        logger.warning(
            "biorxiv_s3: unavailable for %s: %s: %s", doi, type(e).__name__, e
        )
    return None
