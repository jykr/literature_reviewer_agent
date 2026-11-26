"""Moved to the standalone ``paper-sources-mcp`` package
(``packages/paper-sources-mcp``); this name is kept for existing imports and
is the same module object, so patching either name patches both."""

import sys

from paper_sources_mcp import biorxiv_s3 as _module

sys.modules[__name__] = _module
