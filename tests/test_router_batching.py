import threading
import time

import pytest

from scripts.mcp_router import BatchingContextAnswerClient


class _Counter:
    def __init__(self):
        self.n = 0
        self.lock = threading.Lock()

    def inc(self):
        with self.lock:
            self.n += 1
            return self.n


def _fake_call_factory(counter: _Counter):
    def _fake_call(base_url: str, tool: str, args: dict, timeout: float = 1.0):
        # Simulate a tiny network call and count invocations
        counter.inc()
        time.sleep(0.01)
        q = args.get("query")
        queries = args.get("queries") or ([q] if q else ([] if q is None else ([q] if not isinstance(q, list) else q)))
        # When multiple queries are provided (aggregated call), return structured per-query answers
        answers_by_query = None
        if isinstance(q, list) and len(q) > 1:
            answers_by_query = [
                {"query": str(qi), "answer": "ok", "citations": []} for qi in q
            ]
        return {
            "result": {
                "structuredContent": {
                    "result": {
                        "answer": "ok",
                        "citations": [],
                        "query": queries,
                        **({"answers_by_query": answers_by_query} if answers_by_query else {}),
                    }
                }
            }
        }

    return _fake_call


def test_batching_merges_identical_queries():
    counter = _Counter()
    client = BatchingContextAnswerClient(
        call_func=_fake_call_factory(counter),
        enable=True,
        window_ms=120,
        max_batch=8,
        budget_ms=2000,
    )

    results: list[dict] = []
    barrier = threading.Barrier(3)

    def worker():
        barrier.wait()
        res = client.call_or_enqueue(
            "http://localhost:8003/mcp",
            "context_answer",
            {"query": "What is batching?", "limit": 5},
            timeout=1.0,
        )
        results.append(res)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start(); t2.start()
    barrier.wait()  # release both workers
    t1.join(); t2.join()

    # Exactly one underlying call, two client results
    assert counter.n == 1
    assert len(results) == 2
    for r in results:
        assert r.get("result", {}).get("structuredContent", {}).get("result", {}).get("answer") == "ok"


def test_batching_cap_flushes_early():
    counter = _Counter()
    client = BatchingContextAnswerClient(
        call_func=_fake_call_factory(counter),
        enable=True,
        window_ms=5000,  # long window, but cap will force immediate flush
        max_batch=2,
        budget_ms=2000,
    )

    results: list[dict] = []
    barrier = threading.Barrier(3)

    def worker(q):
        barrier.wait()
        res = client.call_or_enqueue(
            "http://localhost:8003/mcp",
            "context_answer",
            {"query": q, "limit": 5},
            timeout=1.0,
        )
        results.append(res)

    t1 = threading.Thread(target=worker, args=("A",))
    t2 = threading.Thread(target=worker, args=("B",))
    t1.start(); t2.start()
    barrier.wait()
    t1.join(); t2.join()

    # Cap reached: we flush once and make a single aggregated call
    assert counter.n == 1
    assert len(results) == 2


def test_bypass_immediate_flag_calls_direct():
    counter = _Counter()
    client = BatchingContextAnswerClient(
        call_func=_fake_call_factory(counter),
        enable=True,
        window_ms=200,
        max_batch=8,
        budget_ms=2000,
    )

    # Two direct calls because of immediate flag; they should not be batched
    r1 = client.call_or_enqueue(
        "http://localhost:8003/mcp",
        "context_answer",
        {"query": "Q1", "limit": 5, "immediate": True},
        timeout=1.0,
    )
    r2 = client.call_or_enqueue(
        "http://localhost:8003/mcp",
        "context_answer",
        {"query": "Q2", "limit": 5, "immediate": True},
        timeout=1.0,
    )

    assert counter.n == 2
    assert r1.get("result", {}).get("structuredContent", {}).get("result", {}).get("answer") == "ok"
    assert r2.get("result", {}).get("structuredContent", {}).get("result", {}).get("answer") == "ok"

