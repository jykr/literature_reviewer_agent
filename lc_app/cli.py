"""CLI for the LangChain/LangGraph literature reviewer.

uv run --extra langchain review-lc "recent single-cell foundation model papers"
uv run --extra langchain review-lc --cv https://example.edu/~me --topics "a; b"
uv run --extra langchain review-lc --model openai:gpt-5 --open "..."
uv run --extra langchain review-lc --focus "clinical genomics" "..."   # or --no-focus
uv run --extra langchain review-lc --topics "a; b" --add 10.1101/2024.01.01.1 --add 2407.06405
    # re-running the same request reuses cached research (.lc_cache/); --refresh to redo
uv run --extra langchain review-lc "DNA language models; also include arxiv.org/abs/2611.27924"
uv run --extra langchain review-lc "add these to my review: https://arxiv.org/abs/2611.24763v1"
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import logging
import os
import re
import webbrowser
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _build_message(cv: str, topics: str) -> str:
    parts = []
    if cv:
        parts.append(f"CV: {cv}")
    if topics:
        parts.append(f"Research topics: {topics}")
    return "\n".join(parts)


def _prompt_inputs() -> str:
    print("\nPaper Triage (LangGraph + paper-sources MCP)\n")
    while True:
        cv = input("CV / website URL (Enter to skip): ").strip()
        topics = input(
            "Research topics / keywords, ';'-separated (Enter to skip): "
        ).strip()
        if msg := _build_message(cv, topics):
            return msg
        print("\nPlease enter at least a URL or one topic (or Ctrl-C to quit).\n")


def since_date(when: str, today: dt.date | None = None) -> str:
    """'6m' / '8w' / '30d' / '1y' back from today, or an ISO date -> ISO date."""
    today = today or dt.date.today()
    if m := re.fullmatch(r"(\d+)\s*([dwmy])", when.strip().lower()):
        n, unit = int(m.group(1)), m.group(2)
        days = {"d": 1, "w": 7, "m": 30.44, "y": 365.25}[unit] * n
        return (today - dt.timedelta(days=round(days))).isoformat()
    try:
        return dt.date.fromisoformat(when.strip()).isoformat()
    except ValueError:
        raise SystemExit(
            f"--added-since: expected a date (2026-05-01) or 6m/8w/30d/1y, got {when!r}"
        ) from None


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a review of recent literature.")
    ap.add_argument(
        "prompt",
        nargs="*",
        help="Free-form request (overrides --cv/--topics). Paper links/ids in it are "
        "added as manual papers; a message of only papers (or 'add these to my "
        "review: ...') adds them to the previous review.",
    )
    ap.add_argument("--cv", default="", help="CV / website URL")
    ap.add_argument(
        "--topics", default="", help="Research topics/keywords, ';'-separated"
    )
    ap.add_argument(
        "--model",
        default=None,
        help="init_chat_model id, e.g. anthropic:claude-sonnet-5-5",
    )
    ap.add_argument(
        "--since",
        default=None,
        help="Earliest publication date YYYY-MM-DD (default: 1 year ago)",
    )
    ap.add_argument(
        "--out", default="literature-reviewer.html", help="Output HTML path"
    )
    ap.add_argument("--open", action="store_true", help="Open the result in a browser")
    ap.add_argument(
        "--focus",
        default=None,
        help="Review lens applied to every stage (default: frontier ML for "
        "genomics/biology; see lc_app/focus.py). Overrides LC_REVIEW_FOCUS.",
    )
    ap.add_argument("--no-focus", action="store_true", help="Disable the review lens.")
    ap.add_argument(
        "--add",
        action="append",
        default=[],
        metavar="ID",
        help="Manually add a paper (DOI, PMID, arXiv id, or URL); repeatable. "
        "Analyzed like searched papers, always included, labeled 'Manually added'.",
    )
    ap.add_argument(
        "--add-file",
        default=None,
        metavar="PATH",
        help="File with one paper identifier per line (# comments allowed).",
    )
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cached research and manual-paper analyses (.lc_cache/).",
    )
    ap.add_argument(
        "--profile",
        default="",
        help="Researcher profile shown in the report and used for relevance, "
        "verbatim (otherwise the model writes one from your input).",
    )
    ap.add_argument(
        "--paperpile-profile",
        nargs="?",
        const="",
        default=None,
        metavar="FOLDER",
        help="Use your Paperpile library (or one folder/label) as an interest "
        "profile, alongside or instead of --cv/--topics.",
    )
    ap.add_argument(
        "--save-to-paperpile",
        nargs="?",
        const="review-lc",
        default=None,
        metavar="LABEL",
        help="Save the reviewed papers to Paperpile under LABEL (default "
        "'review-lc'); existing copies just gain the label.",
    )
    ap.add_argument(
        "--include-paperpile",
        default=None,
        metavar="LABEL",
        help="Add every paper in a Paperpile folder/label (e.g. 'to read') to the "
        "review, shown in their own color.",
    )
    ap.add_argument(
        "--added-since",
        default=None,
        metavar="WHEN",
        help="With --include-paperpile: only papers added to the library since "
        "WHEN, a date (2026-05-01) or a span back from today (6m, 8w, 30d, 1y).",
    )
    ap.add_argument(
        "--only-added",
        action="store_true",
        help="Skip the literature search: review only added papers "
        "(--add, pasted links, --include-paperpile).",
    )
    ap.add_argument(
        "--no-paperpile",
        action="store_true",
        help="Skip Paperpile even when PAPERPILE_API_KEY is set.",
    )
    args = ap.parse_args()

    if args.model:
        os.environ["LC_REVIEW_MODEL"] = args.model
    if args.no_focus:
        os.environ["LC_REVIEW_FOCUS"] = "none"
    elif args.focus is not None:
        os.environ["LC_REVIEW_FOCUS"] = args.focus
    message = " ".join(args.prompt) or _build_message(args.cv, args.topics)
    message = message or args.profile  # a profile alone is enough to scope
    if not message and args.only_added:
        message = " "  # scope comes from the added papers themselves
    if not message and args.paperpile_profile is not None:
        message = "Recent papers in the areas my Paperpile library covers."
    message = message or _prompt_inputs()
    use_paperpile = not args.no_paperpile and bool(
        os.getenv("PAPERPILE_API_KEY")
        or args.paperpile_profile is not None
        or args.save_to_paperpile
        or args.include_paperpile
    )

    from lc_app.graph import run_review  # after env/model setup
    from lc_app.intake import parse_request

    # Root at WARNING, our pipeline at INFO: library loggers (httpx, anthropic,
    # ADK) reconfigure their own levels on lazy import, so filter by origin.
    logging.basicConfig(level=logging.WARNING, format="▶ %(message)s", force=True)
    logging.getLogger("lc_app").setLevel(logging.INFO)

    manual_ids = list(args.add)
    if args.add_file:
        for line in Path(args.add_file).read_text(encoding="utf-8").splitlines():
            if line := line.split("#", 1)[0].strip():
                manual_ids.append(line)

    async def go():
        parsed = await parse_request(message)
        ids = list(dict.fromkeys(parsed.manual_ids + manual_ids))
        if parsed.manual_ids:
            logging.getLogger("lc_app.intake").info(
                "intake: %d paper(s) from the request: %s; %s",
                len(parsed.manual_ids),
                ", ".join(parsed.manual_ids),
                "adding to previous review" if parsed.add_to_previous else "new review",
            )
        return await run_review(
            parsed.request,
            output_path=args.out,
            from_date=args.since,
            manual_ids=ids,
            refresh=args.refresh,
            add_to_previous=parsed.add_to_previous,
            paperpile=use_paperpile,
            paperpile_profile=args.paperpile_profile if use_paperpile else None,
            save_label=args.save_to_paperpile if use_paperpile else None,
            include_collection=args.include_paperpile if use_paperpile else None,
            skip_research=args.only_added,
            added_since=since_date(args.added_since) if args.added_since else None,
            profile=args.profile,
        )

    try:
        final = asyncio.run(go())
    except ValueError as e:
        raise SystemExit(f"\n{e}") from None
    print("\n" + (final.get("message") or "(no output)"))
    out = Path(args.out).resolve()
    if args.open and final.get("queue", {}).get("papers") and out.exists():
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
