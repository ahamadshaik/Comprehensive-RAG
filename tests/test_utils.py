from app.utils import tokenize_simple


def test_tokenize_simple_lowercases() -> None:
    assert tokenize_simple("Hello World") == ["hello", "world"]
