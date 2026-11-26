"""LangChain/LangGraph variant of the literature reviewer.

Same pipeline and output as the ADK app in ``app/`` (scope -> parallel research
-> rank -> render), but orchestrated with LangGraph, with the model chosen via
``init_chat_model``, and with paper search served by the standalone ``paper-sources-mcp`` server
(``packages/paper-sources-mcp``) that pools OpenAlex, Europe PMC, PubMed, arXiv
and bioRxiv/medRxiv.

Kept import-free on purpose so importing ``lc_app.*`` never pulls in the ADK
app.
"""
