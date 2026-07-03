#!/usr/bin/env python3
"""Progressive training evaluation - measures quality at checkpoints."""
import sys, os
sys.path.insert(0, '.')

from scripts.rerank_eval import get_candidates, rerank_learning, rerank_onnx, DEFAULT_EVAL_QUERIES
from scripts.rerank_recursive import rerank_with_learning
from scripts.learning_reranker_worker import CollectionLearner
import numpy as np

EXTRA_QUERIES = [
    'function python', 'class MCP', 'vector embedding', 'cache LRU',
    'subprocess async', 'tokenizer ONNX', 'collection Qdrant', 'rerank fusion',
    'memory store', 'config env', 'error exception', 'test pytest',
    'docker build', 'API handler', 'database pool',
]

def measure_quality(collection='eval'):
    """Measure MRR against ONNX ground truth."""
    eval_qs = DEFAULT_EVAL_QUERIES[:6]
    mrrs = []
    for eq in eval_qs:
        cands = get_candidates(eq, limit=20)
        if not cands:
            continue
        onnx_ranked = rerank_onnx(eq, [c.copy() for c in cands])
        onnx_top5 = set(c['path'] for c in onnx_ranked[:5])
        learn_ranked = rerank_learning(eq, [c.copy() for c in cands], collection=collection)
        for rank, c in enumerate(learn_ranked, 1):
            if c['path'] in onnx_top5:
                mrrs.append(1.0 / rank)
                break
        else:
            mrrs.append(0.0)
    return np.mean(mrrs) if mrrs else 0

def main():
    all_queries = (DEFAULT_EVAL_QUERIES + EXTRA_QUERIES) * 15
    checkpoints = [50, 100, 150, 200, 250, 300]
    results = []
    query_count = 0
    
    print('Queries | Samples | Loss   | MRR   | Distill%', flush=True)
    print('-' * 50, flush=True)
    
    for query in all_queries:
        candidates = get_candidates(query, limit=25)
        if candidates:
            rerank_with_learning(query, candidates, learn_from_onnx=True, collection='eval')
            query_count += 1
        
        if query_count in checkpoints:
            # Clear stale locks and process events
            import glob
            for lock in glob.glob('/tmp/rerank_weights/eval_*.lock'):
                try:
                    os.remove(lock)
                except Exception:
                    pass

            learner = CollectionLearner(collection='eval')
            learner.process_events()
            m = learner.scorer.get_metrics()
            
            # Measure quality
            mrr = measure_quality()
            
            print(f'{query_count:7} | {m.get("total_samples",0):7} | {m.get("avg_loss",0):6.3f} | {mrr:.3f} | {mrr*100:.1f}%', flush=True)
            results.append({
                'queries': query_count,
                'samples': m.get('total_samples', 0),
                'loss': round(float(m.get('avg_loss', 0)), 3),
                'mrr': round(mrr, 3),
                'distill_pct': round(mrr * 100, 1),
            })
            checkpoints.remove(query_count)
        
        if query_count >= 300:
            break
    
    print('\n' + '=' * 50)
    print('PROGRESSIVE TRAINING RESULTS')
    print('=' * 50)
    for r in results:
        print(f"  {r['queries']:3} queries → MRR {r['mrr']:.3f} ({r['distill_pct']}% of ONNX)")
    
    return results

if __name__ == '__main__':
    main()

