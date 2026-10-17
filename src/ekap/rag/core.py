"""In-memory RAG: ingest documents, retrieve ranked chunks, cite sources.

Pure Python token-overlap scoring. No network, embeddings, or external store.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(_TOKEN.findall(text.lower()))
