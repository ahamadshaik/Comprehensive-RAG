import pytest
from pydantic import ValidationError

from app.config import Settings


def test_fusion_weights_must_sum_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BM25_WEIGHT", "0.3")
    monkeypatch.setenv("DENSE_WEIGHT", "0.3")
    with pytest.raises(ValidationError, match="sum to 1"):
        Settings()
