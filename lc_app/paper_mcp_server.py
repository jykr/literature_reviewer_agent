"""Moved to the standalone ``paper-sources-mcp`` package
(``packages/paper-sources-mcp``); this name is kept for existing imports and
``python -m lc_app.paper_mcp_server``."""

import sys

from paper_sources_mcp import server as _module

if __name__ == "__main__":
    _module.main()
else:  # the same module object under the old name
    sys.modules[__name__] = _module
