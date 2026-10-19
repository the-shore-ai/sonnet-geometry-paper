"""Separate poem susceptibility from shared responses to individual scrambles."""
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import rankdata

from analyze_signal_audit import CONTROLS, receipt, finite_summary
from sonnet_geometry.confirmation_analysis import SEEDS
from sonnet_geometry.signal_audit import rank_correlation
from sonnet_geometry.storage import canonical_json, write_exclusive
from sonnet_geometry.text_control_analysis import FAMILIES, STATES, POOLS


def main():
    out = Path('outputs/signal-audit-v6/iteration-5-response-components.json')
    if out.exists():
        raise FileExistsError(out)
    start = time.monotonic()
    first_path = Path('outputs/signal-audit-v6/first-pass.json')
    first = json.loads(first_path.read_text())
    rng = np.random.default_rng(2026101922)
    permutations = np.stack([[rng.permutation(5) for _ in range(24)] for _ in range(199)])
    rows = []
    for pop in first['populations']:
        index = {(r['model'], r['precision'], r['intervention'], r['state'], r['pooling'], r['seed']): r
                 for r in pop['paired']}
        for left, right in combinations(FAMILIES, 2):
            for precision in ('fp32', 'bf16'):
                for intervention in CONTROLS:
                    for state in STATES[1:]:
                        for pool in POOLS:
                            a, b = [np.stack([index[f, precision, intervention, state, pool, s]['per_poem_cosine_displacement']
                                             for s in SEEDS], axis=1) for f in (left, right)]
                            cross = [[rank_correlation(a[:, i], b[:, j]) for j in range(5)] for i in range(5)]
                            diagonal = [cross[i][i] for i in range(5)]
                            other = [cross[i][j] for i in range(5) for j in range(5) if i != j]
                            ar, br = [rankdata(np.round(v, 12), axis=1) for v in (a, b)]
                            within = rank_correlation(ar.ravel(), br.ravel())
                            calibrated = [rank_correlation(ar.ravel(), np.take_along_axis(br, p, axis=1).ravel()) for p in permutations]
                            rows.append({'population': pop['population'], 'models': [left, right], 'precision': precision,
                                         'intervention': intervention, 'state': state, 'pooling': pool,
                                         'cross_seed_rhos': cross, 'same_seed': finite_summary(diagonal),
                                         'different_seed': finite_summary(other),
                                         'mean_across_seeds_poem_response_rho': rank_correlation(a.mean(1), b.mean(1)),
                                         'within_poem_seed_rank_correlation': within,
                                         'per_poem_seed_rank_rhos': [rank_correlation(x, y) for x, y in zip(ar, br, strict=True)],
                                         'within_poem_seed_permutation_reference': calibrated,
                                         'within_poem_seed_permutation_summary': finite_summary(calibrated)})
    result = {'schema_version': 'signal-audit-response-components-v6',
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'decision': receipt('experiments/signal-audit-v6/iteration-5-decision.json'),
              'first_pass': receipt(first_path), 'code': [receipt(__file__), receipt('src/sonnet_geometry/signal_audit.py')],
              'rows': rows, 'within_poem_seed_permutations': permutations.tolist(),
              'limits': 'This decomposition follows inspecting the full response grid. Means, matched/mismatched seed '
                        'contrasts and within-poem ranks answer different descriptive questions. Independent within-poem '
                        'seed permutations calibrate realization matching without changing poem susceptibility; no p-values '
                        'or literary mechanism are claimed.', 'new_model_calls': 0, 'elapsed_seconds': time.monotonic() - start}
    write_exclusive(out, canonical_json(result))
    print(json.dumps({'complete': True, 'rows': len(rows), 'output': receipt(out), 'seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
