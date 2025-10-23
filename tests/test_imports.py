def test_import_public_modules() -> None:
    import app.config  # noqa: F401
    import app.ingest  # noqa: F401
    import app.retriever  # noqa: F401
    import app.rag  # noqa: F401
