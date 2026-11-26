"""Paperpile's remote MCP server (beta), signed in with OAuth.

https://mcp.paperpile.com/ speaks streamable HTTP and advertises OAuth 2.1
(dynamic client registration + PKCE) on api.paperpile.com. The MCP SDK's
``OAuthClientProvider`` runs that flow; this module supplies a browser
redirect, a one-shot localhost callback, and token storage in
``<LC_CACHE_DIR>/paperpile/`` (owner-only file permissions).

    uv run --extra langchain python -m lc_app.paperpile   # connect + list tools

With ``PAPERPILE_API_KEY`` set (in .env) it is sent as a Bearer token and no
browser sign-in happens. Otherwise the stored OAuth token is reused after the
first sign-in; delete the folder to sign out.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

from .cache import cache_dir

SERVER_URL = os.getenv("PAPERPILE_MCP_URL", "https://mcp.paperpile.com/")
CALLBACK_PORT = int(os.getenv("PAPERPILE_CALLBACK_PORT", "8765"))
REDIRECT_URI = f"http://127.0.0.1:{CALLBACK_PORT}/callback"


class FileTokenStorage(TokenStorage):
    """OAuth client registration + tokens as JSON files, mode 0600."""

    def __init__(self, folder: Path | None = None):
        self.folder = folder or cache_dir() / "paperpile"

    def _read(self, name: str) -> dict | None:
        try:
            return json.loads((self.folder / name).read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def _write(self, name: str, payload: dict) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        os.chmod(self.folder, 0o700)
        path = self.folder / name
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)

    async def get_tokens(self) -> OAuthToken | None:
        data = self._read("tokens.json")
        return OAuthToken.model_validate(data) if data else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self._write("tokens.json", tokens.model_dump(mode="json", exclude_none=True))

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        data = self._read("client.json")
        return OAuthClientInformationFull.model_validate(data) if data else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self._write(
            "client.json", client_info.model_dump(mode="json", exclude_none=True)
        )


async def _open_browser(url: str) -> None:
    print(f"Opening Paperpile sign-in in your browser:\n  {url}", file=sys.stderr)
    webbrowser.open(url)


async def _wait_for_callback() -> tuple[str, str | None]:
    """Serve one request on REDIRECT_URI and return (code, state)."""
    result: dict[str, Any] = {}
    done = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            if url.path != "/callback":
                self.send_response(404)
                self.end_headers()
                return
            result.update({k: v[0] for k, v in parse_qs(url.query).items()})
            ok = "code" in result
            self.send_response(200 if ok else 400)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"Paperpile connected; you can close this tab."
                if ok
                else b"Paperpile sign-in failed; see the terminal."
            )
            done.set()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", CALLBACK_PORT), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        await asyncio.to_thread(done.wait, 900)
    finally:
        server.shutdown()
    if "code" not in result:
        reason = result.get("error", "timed out")
        if detail := result.get("error_description"):
            reason += f" ({detail})"
        raise RuntimeError(f"Paperpile sign-in failed: {reason}")
    return result["code"], result.get("state")


def auth_provider(storage: TokenStorage | None = None) -> OAuthClientProvider:
    return OAuthClientProvider(
        server_url=SERVER_URL,
        client_metadata=OAuthClientMetadata(
            client_name="literature-reviewer (review-lc)",
            redirect_uris=[REDIRECT_URI],
            grant_types=["authorization_code"],
            response_types=["code"],
            token_endpoint_auth_method="none",
        ),
        storage=storage or FileTokenStorage(),
        redirect_handler=_open_browser,
        callback_handler=_wait_for_callback,
        timeout=900.0,
    )


def paperpile_connection() -> dict[str, Any]:
    """MultiServerMCPClient connection for Paperpile: API key if set, else OAuth."""
    if key := os.getenv("PAPERPILE_API_KEY", "").strip():
        return {
            "transport": "streamable_http",
            "url": SERVER_URL,
            "headers": {"Authorization": f"Bearer {key}"},
        }
    return {"transport": "streamable_http", "url": SERVER_URL, "auth": auth_provider()}


# ---------------------------------------------------------------------------
# Library operations used by the review (called from code, never by the LLM)
# ---------------------------------------------------------------------------

_PROFILE_LIMIT = 100
_RATIONALE = "literature-reviewer (review-lc): {}"


def _norm_doi(doi: str) -> str:
    doi = (doi or "").strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        doi = doi.removeprefix(prefix)
    return doi


def arxiv_doi(arxiv_id: str) -> str:
    """arXiv's DataCite DOI, version suffix dropped: 2608.24688v2 -> 10.48550/arxiv.2608.24688."""
    base = re.sub(r"v\d+$", "", arxiv_id.split(":")[-1])
    return f"10.48550/arxiv.{base.lower()}"


def profile_text(refs: list[dict], where: str) -> str:
    """Titles/venues/years of ``refs`` as an interest profile for scoping."""
    lines = []
    for ref in refs:
        year = (ref.get("published") or "")[:4]
        venue = ref.get("journal") or ""
        lines.append(
            "- " + " · ".join(x for x in (ref.get("title", ""), venue, year) if x)
        )
    if not lines:
        return ""
    return (
        f"Papers in the researcher's Paperpile {where} ({len(lines)} shown); "
        "treat them as their reading interests:\n" + "\n".join(lines)
    )


class PaperpileLibrary:
    """Thin wrapper over an open Paperpile MCP session."""

    def __init__(self, session):
        self.session = session

    async def call(self, tool: str, why: str, **args) -> dict:
        res = await self.session.call_tool(
            tool, {"rationale": _RATIONALE.format(why), "output_format": "JSON", **args}
        )
        text = "".join(getattr(c, "text", "") for c in res.content)
        if res.isError:
            raise RuntimeError(f"Paperpile {tool} failed: {text[:300]}")
        return json.loads(text) if text.strip() else {}

    async def collections(self) -> list[dict]:
        data = await self.call(
            "get_collections_from_library",
            "find a folder or label by name",
            include_shared_libraries=False,
        )
        return data.get("data", [])

    async def find_collection(self, name: str) -> dict | None:
        want = name.strip().lower()
        return next(
            (c for c in await self.collections() if c["name"].strip().lower() == want),
            None,
        )

    async def lookup(self, dois: list[str]) -> list[dict]:
        """Library references matching any of ``dois`` (batches of 100)."""
        found: list[dict] = []
        uniq = sorted({_norm_doi(d) for d in dois if d})
        for i in range(0, len(uniq), 100):
            data = await self.call(
                "get_references_from_library",
                "flag reviewed papers already in the user's library",
                dois=uniq[i : i + 100],
            )
            found.extend(data.get("data", []))
        return found

    async def collection_refs(self, name: str) -> tuple[str, list[dict]]:
        """(exact name, all non-trashed references) of a folder or label."""
        coll = await self.find_collection(name)
        if coll is None:
            names = ", ".join(c["name"] for c in await self.collections())
            raise ValueError(
                f"No Paperpile folder or label named {name!r}. Available: {names}"
            )
        refs: list[dict] = []
        cursor = None
        while True:
            args: dict[str, Any] = {"collection_id": coll["id"], "limit": 100}
            if cursor:
                args["cursor"] = cursor
            data = await self.call(
                "get_references_from_library",
                f"add the papers in '{coll['name']}' to a literature review",
                **args,
            )
            refs.extend(data.get("data", []))
            cursor = data.get("cursor")
            if not cursor or not data.get("data"):
                return coll["name"], refs

    async def profile(self, collection: str | None = None) -> str:
        """Interest profile from up to 100 library papers (or one folder/label)."""
        args: dict[str, Any] = {"limit": _PROFILE_LIMIT}
        where = "library"
        if collection:
            coll = await self.find_collection(collection)
            if coll is None:
                names = ", ".join(c["name"] for c in await self.collections())
                raise ValueError(
                    f"No Paperpile folder or label named {collection!r}. "
                    f"Available: {names}"
                )
            args["collection_id"] = coll["id"]
            where = f"folder/label '{coll['name']}'"
        data = await self.call(
            "get_references_from_library",
            "use the user's papers as an interest profile to scope a review",
            **args,
        )
        return profile_text(data.get("data", []), where)

    async def ensure_label(self, name: str) -> str:
        coll = await self.find_collection(name)
        if coll:
            return coll["id"]
        data = await self.call(
            "update_collections",
            "create a label for papers picked by a literature review",
            updates=[{"action": "create", "type": "label", "name": name}],
        )
        coll = await self.find_collection(name)
        if coll is None:
            raise RuntimeError(f"Could not create Paperpile label {name!r}: {data}")
        return coll["id"]

    async def save(self, dois: list[str], label: str) -> dict[str, int]:
        """Add ``dois`` to the library under ``label``; label existing copies too.

        Never trashes, deletes or edits notes/metadata: new references are
        created, and already-present ones only gain the label.
        """
        label_id = await self.ensure_label(label)
        dois = sorted({_norm_doi(d) for d in dois if d})
        before = {_norm_doi(r.get("doi", "")) for r in await self.lookup(dois)}
        new = [d for d in dois if d not in before]
        for i in range(0, len(new), 50):
            await self.call(
                "save_to_library",
                "save papers picked by a literature review",
                dois=new[i : i + 50],
                labels=[label_id],
                skip_duplicates=True,
                find_pdfs=True,
                wait_for_pdf=False,
            )
        refs = await self.lookup(dois)
        updates = [
            {"id": r["id"], "add_labels": [label_id]}
            for r in refs
            if label_id not in (r.get("labels") or [])
        ]
        for i in range(0, len(updates), 50):
            await self.call(
                "update_references",
                "label existing library copies of papers picked by a review",
                updates=updates[i : i + 50],
            )
        saved = {_norm_doi(r.get("doi", "")) for r in refs}
        return {
            "added": len([d for d in new if d in saved]),
            "labeled_existing": len(before & saved),
            "failed": len([d for d in dois if d not in saved]),
        }


async def _main() -> None:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools

    client = MultiServerMCPClient({"paperpile": paperpile_connection()})
    async with client.session("paperpile") as session:
        for tool in await load_mcp_tools(session):
            schema = tool.args_schema if isinstance(tool.args_schema, dict) else {}
            print(f"\n## {tool.name}\n{tool.description}")
            print("args:", json.dumps(schema.get("properties", {}), indent=1))


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    asyncio.run(_main())
