from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))            # legacy.py, cases.py
sys.path.insert(0, str(HERE.parent))     # the horus package

from legacy import REPO, FakeEmbedder, FakeGemini, load_legacy  # noqa: E402


class FakeLLM:
    """New-engine backend that returns the same queued replies FakeGemini does."""

    name = "fake"

    def __init__(self):
        self.replies: list = []
        self.calls: list = []

    def generate(self, prompt, *, system=None, temperature=0.0, max_tokens=1024, json_output=False,
                 schema=None):
        self.calls.append({"prompt": prompt, "system": system, "temperature": temperature,
                           "max_tokens": max_tokens, "json_output": json_output})
        if not self.replies:
            from horus.llm import LLMUnavailable
            raise LLMUnavailable("no reply queued")
        return self.replies.pop(0)


@pytest.fixture(scope="session", autouse=True)
def _isolated_dirs(tmp_path_factory):
    """Demo code reads data files from the working directory; the engine uses Paths.
    Point both at the repository's data, and keep user data in a temp folder."""
    user = tmp_path_factory.mktemp("horus_user")
    os.environ["HORUS_RESOURCE_DIR"] = str(REPO)
    os.environ["HORUS_USER_DIR"] = str(user)
    old = os.getcwd()
    os.chdir(REPO)
    yield
    os.chdir(old)


@pytest.fixture(scope="session")
def legacy():
    return load_legacy()


@pytest.fixture
def gemini():
    FakeGemini.replies = []
    FakeGemini.calls = []
    yield FakeGemini
    FakeGemini.replies = []


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture(scope="session")
def paths():
    from horus.config import Paths
    return Paths()


def make_engine(paths, semantic: bool, llm=None, disk_cache=False):
    from horus.engine import HorusEngine
    return HorusEngine(paths, embedder=FakeEmbedder() if semantic else None, llm=llm,
                       use_disk_cache=disk_cache)
