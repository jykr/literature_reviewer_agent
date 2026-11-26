"""LangChain/LangGraph variant of the literature reviewer.

Same pipeline and output as the ADK app in ``app/`` (scope -> parallel research
-> rank -> render), but orchestrated with LangGraph, with the model chosen via
``init_chat_model``, and with paper search served by a local MCP server
(``lc_app.paper_mcp_server``) that pools OpenAlex, Europe PMC, PubMed, arXiv
and bioRxiv/medRxiv.

Kept import-free on purpose: the MCP server subprocess imports
``lc_app.paper_sources`` and must not pull in the ADK app.
"""
