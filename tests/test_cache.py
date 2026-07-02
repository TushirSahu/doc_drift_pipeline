import time

from src.core.cache import TTLCache, make_key


def test_set_and_get():
    c = TTLCache(maxsize=4, ttl=100)
    c.set("k", [1, 2, 3])
    assert c.get("k") == [1, 2, 3]


def test_miss_returns_none():
    c = TTLCache()
    assert c.get("absent") is None


def test_ttl_expiry():
    c = TTLCache(maxsize=4, ttl=0.05)
    c.set("k", "v")
    assert c.get("k") == "v"
    time.sleep(0.06)
    assert c.get("k") is None


def test_lru_eviction():
    c = TTLCache(maxsize=2, ttl=100)
    c.set("a", 1)
    c.set("b", 2)
    c.get("a")          # touch a → b is now least-recently-used
    c.set("c", 3)       # evicts b
    assert c.get("b") is None
    assert c.get("a") == 1
    assert c.get("c") == 3


def test_get_or_compute_caches():
    c = TTLCache()
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return 42

    assert c.get_or_compute("k", compute) == 42
    assert c.get_or_compute("k", compute) == 42
    assert calls["n"] == 1  # computed once


def test_stats_track_hits_and_misses():
    c = TTLCache()
    c.get("x")          # miss
    c.set("x", 1)
    c.get("x")          # hit
    stats = c.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


def test_make_key_is_stable_and_distinct():
    assert make_key("a", 1) == make_key("a", 1)
    assert make_key("a", 1) != make_key("a", 2)
