from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from torch import nn
from torchvision.models import resnet18

from app.config import get_settings
from app.main import create_app


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FastAPI]:
    model_path = tmp_path / "classifier.pt"
    base_model = resnet18(weights=None)
    base_model.fc = nn.Linear(base_model.fc.in_features, 1)
    torch.save(base_model.state_dict(), model_path)

    monkeypatch.setenv("TEMP_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("USE_GPU", "false")
    get_settings.cache_clear()
    test_app = create_app()
    yield test_app
    get_settings.cache_clear()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
