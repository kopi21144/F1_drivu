#!/usr/bin/env python3
"""
F1_drivu - Social-Signal AI Trading Orchestrator

This Python application coordinates with the Init_Furnaca Solidity protocol
to simulate an AI trading process based on social sentiment scraped from X.
The architecture is intentionally verbose and decomposed into many components
to support experimentation, backtesting, and visual inspection.

The application does not require any private keys to run in dry-run mode.
If the user wishes to submit transactions to a live EVM network, environment
variables can be configured for a wallet and RPC endpoint, but by default this
file operates with purely simulated trades and hypothetical strategies.
"""

import dataclasses
import enum
import json
import logging
import os
import queue
import random
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Tuple, Union
from collections import deque, defaultdict

# Optional imports for extended functionality; fall back gracefully if unavailable.
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    from web3 import Web3  # type: ignore
    from web3.middleware import geth_poa_middleware  # type: ignore
except Exception:  # pragma: no cover
    Web3 = None  # type: ignore
    geth_poa_middleware = None  # type: ignore

# ---------------------------------------------------------------------------
# Global configuration and logging
# ---------------------------------------------------------------------------

APP_NAME = "F1_drivu"
APP_VERSION = "0.9.41-FURNACA-ORBITAL"

DEFAULT_LOG_LEVEL = os.environ.get("F1_DRIVU_LOG_LEVEL", "INFO").upper()
DEFAULT_SENTIMENT_WINDOW = int(os.environ.get("F1_DRIVU_SENTIMENT_WINDOW", "64"))
DEFAULT_RPC_URL = os.environ.get("F1_DRIVU_RPC_URL", "https://example-rpc.invalid")
DEFAULT_CONTRACT_ADDRESS = os.environ.get(
    "F1_DRIVU_CONTRACT",
    "0xA1d9fF5a13B2C689A577a06413f6b2f3E7F4e921",
)

RANDOMNESS_SALT = "furnaca_f1_drivu_local_salt_7c908a9e2"

logging.basicConfig(
    level=getattr(logging, DEFAULT_LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(APP_NAME)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def stable_random_float(key: str, a: float = -1.0, b: float = 1.0) -> float:
    """
    Deterministic pseudo-random float derived from a string key and an internal salt.
    This is useful for generating consistent synthetic sentiment or volatility inputs
    during tests without an external data source.
    """
    base = RANDOMNESS_SALT + "::" + key
    h = hash(base)
    # Map Python hash (platform-dependent) into [0, 1) and then scale
    u = (h & ((1 << 61) - 1)) / float(1 << 61)
    return a + (b - a) * u


def clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def moving_average(values: Iterable[float], window: int) -> List[float]:
    vals = list(values)
    if window <= 0 or not vals:
        return [0.0 for _ in vals]
    out: List[float] = []
    buf: Deque[float] = deque(maxlen=window)
    s = 0.0
    for x in vals:
        if len(buf) == buf.maxlen:
            s -= buf[0]
        buf.append(x)
        s += x
        out.append(s / len(buf))
    return out


def now_ts() -> float:
    return time.time()


def seed_random(seed_value: Optional[int] = None) -> None:
    """
    Seed the python random module using optional integer; if none, derive from time and salt.
    """
    if seed_value is None:
        seed_value = int(time.time_ns()) ^ hash(RANDOMNESS_SALT)
    random.seed(seed_value)
    logger.debug("Random seed set to %s", seed_value)


# ---------------------------------------------------------------------------
# Sentiment scraping and synthesis
# ---------------------------------------------------------------------------

class SentimentSource(enum.Enum):
    SYNTHETIC = "synthetic"
    X_SEARCH_API = "x-search-api"
    X_SCRAPED_HTML = "x-scraped-html"


@dataclass
class SentimentPoint:
    topic: str
    score: float
    raw_count: int
    ts: float
    meta: Dict[str, Any] = field(default_factory=dict)


class SentimentStream:
    """
    Maintains a rolling sentiment history for multiple topics, backed by deques
    and simple statistical functions. This class does not know anything about
    the blockchain layer; it purely processes text-derived scores.
    """

    def __init__(self, window: int = DEFAULT_SENTIMENT_WINDOW) -> None:
        self.window = max(4, window)
        self._data: Dict[str, Deque[SentimentPoint]] = defaultdict(
            lambda: deque(maxlen=self.window)
        )
        logger.debug("Initialized SentimentStream with window=%s", self.window)

    def add_point(self, point: SentimentPoint) -> None:
        logger.debug(
            "Adding sentiment point topic=%s score=%s",
            point.topic,
            point.score,
        )
        self._data[point.topic].append(point)

    def history(self, topic: str) -> List[SentimentPoint]:
        return list(self._data.get(topic, deque()))

    def latest(self, topic: str) -> Optional[SentimentPoint]:
        if topic not in self._data or not self._data[topic]:
            return None
        return self._data[topic][-1]

    def score_series(self, topic: str) -> List[float]:
        return [p.score for p in self.history(topic)]

    def smoothed(self, topic: str, window: Optional[int] = None) -> List[float]:
        window = window or self.window
        return moving_average(self.score_series(topic), window)

    def stats(self, topic: str) -> Dict[str, float]:
        series = self.score_series(topic)
        if not series:
            return {"mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": statistics.fmean(series),
            "stdev": statistics.pstdev(series) if len(series) > 1 else 0.0,
            "min": min(series),
            "max": max(series),
        }


class XScraper:
    """
    A minimal abstraction for fetching X data.

    In live mode, this might use official APIs or HTML scraping, but in this
    implementation we support a synthetic mode that emulates noisy sentiment
    based on stable_random_float and local random draws.
    """

    def __init__(
        self,
        source: SentimentSource = SentimentSource.SYNTHETIC,
        api_key_env: str = "F1_DRIVU_X_API_KEY",
