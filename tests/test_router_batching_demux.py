import threading
import time

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
        counter.inc()
        time.sleep(0.01)
        q = args.get("query")
        queries = args.get("queries") or ([q] if q else ([] if q is None else ([q] if not isinstance(q, list) else q)))
        answers_by_query = None
        if isinstance(q, list) and len(q) > 1:
            answers_by_query = [
                {"query": str(qi), "answer": f"ok:{qi}", "citations": []} for qi in q
            ]
        return {
            "result": {
                "structuredContent": {
                    "result": {
                        "answer": f"ok:{q}",
                        "citations": [],
                        "query": queries,
                        **({"answers_by_query": answers_by_query} if answers_by_query else {}),
                    }
                }
            }
        }

    return _fake_call


def test_demultiplex_different_queries_results_are_isolated():
    counter = _Counter()
    client = BatchingContextAnswerClient(
        call_func=_fake_call_factory(counter),
        enable=True,
        window_ms=120,
        max_batch=8,
        budget_ms=2000,
    )

    results: list[tuple[str, dict]] = []
    barrier = threading.Barrier(3)

    def worker(q: str):
        barrier.wait()
        res = client.call_or_enqueue(
            "http://localhost:8003/mcp",
            "context_answer",
            {"query": q, "limit": 5},
            timeout=1.0,
        )
        results.append((q, res))

    t1 = threading.Thread(target=worker, args=("Q1",))
    t2 = threading.Thread(target=worker, args=("Q2",))
    t1.start(); t2.start()
    barrier.wait()
    t1.join(); t2.join()

    # Aggregated call once, demux per-query reply
    assert counter.n == 1
    assert len(results) == 2
    for q, r in results:
        rq = r.get("result", {}).get("structuredContent", {}).get("result", {}).get("query")
        # Each result should reflect only its own query
        assert rq == [q]


def test_budget_fallback_does_not_double_call():
    counter = _Counter()
    client = BatchingContextAnswerClient(
        call_func=_fake_call_factory(counter),
        enable=True,
        window_ms=500,   # long window so timer would fire later
        max_batch=8,
        budget_ms=10,    # tiny budget to force immediate fallback
    )

    res = client.call_or_enqueue(
        "http://localhost:8003/mcp",
        "context_answer",
        {"query": "late", "limit": 5},
        timeout=1.0,
    )
    assert res
    # Wait beyond the window; if slot was not removed, we'd see a second call when timer flushes
    time.sleep(0.6)
    assert counter.n == 1

