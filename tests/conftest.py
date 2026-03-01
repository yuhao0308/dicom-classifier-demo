from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FastAPI]:
    monkeypatch.setenv("TEMP_DIR", str(tmp_path))
    get_settings.cache_clear()
    test_app = create_app()
    yield test_app
    get_settings.cache_clear()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
