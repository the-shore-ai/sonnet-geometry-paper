"""Iteration 3: separate recoverable identity from preserved relational geometry."""
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from safetensors import safe_open

from analyze_signal_audit import POPULATIONS, CONTROLS, get_matrix, receipt
from sonnet_geometry.confirmation_analysis import condition_name, SEEDS
from sonnet_geometry.signal_audit import rank_correlation, partial_rank, rdm, unit_rows, upper
from sonnet_geometry.storage import canonical_json, write_exclusive
from sonnet_geometry.text_control_analysis import FAMILIES, STATES


def euclidean_square(x):
    norms = np.sum(x * x, axis=1)
    distances = np.maximum(0, norms[:, None] + norms[None, :] - 2 * (x @ x.T))
    np.fill_diagonal(distances, 0)
    return distances


def identity_calibration(a, b, permutations):
    scores = np.round(unit_rows(a) @ unit_rows(b).T, 12)
    rows = scores == scores.max(axis=1, keepdims=True)
    cols = scores == scores.max(axis=0, keepdims=True)
    credit = (rows / rows.sum(axis=1, keepdims=True) + cols / cols.sum(axis=0, keepdims=True)) / 2
    actual = float(np.trace(credit) / len(credit))
    reference = [float(credit[np.arange(len(credit)), p].mean()) for p in permutations]
    return {'top1': actual, 'chance_top1': 1 / len(credit), 'permuted_identity_top1': reference}


def main():
    out = Path('outputs/signal-audit-v6/iteration-3-geometry.json')
    if out.exists():
        raise FileExistsError(out)
    start = time.monotonic()
    studies = {p: json.loads(Path(path).read_text()) for p, path in POPULATIONS.items()}
    rng = np.random.default_rng(2026101921)
    permutations = [rng.permutation(24) for _ in range(199)]
    rows, negatives = [], []
    for family in FAMILIES:
        with ExitStack() as stack:
            handles = {}
            for pop, study in studies.items():
                pooled = next(x for x in study['models'][family]['source_files'] if x['path'].endswith('.safetensors'))
                assert receipt(pooled['path'])['sha256'] == pooled['sha256']
                handles[pop] = stack.enter_context(safe_open(pooled['path'], framework='numpy'))
            for precision in ('fp32', 'bf16'):
                for state in STATES[1:]:
                    data = {}
                    for pop, study in studies.items():
                        h, ids = handles[pop], study['poems']
                        x = get_matrix(h, ids, condition_name('original', None, precision), state, 'mean')
                        ys = {c: [get_matrix(h, ids, condition_name(c, seed, precision), state, 'mean') for seed in SEEDS]
                              for c in CONTROLS}
                        input_rdm = rdm(get_matrix(h, ids, condition_name('original', None, precision), 'input', 'mean'))
                        data[pop] = x, ys, input_rdm
                    for calibration, test in [('sidney', 'spenser'), ('spenser', 'sidney')]:
                        train_x, train_ys, _ = data[calibration]
                        x, ys, input_rdm = data[test]
                        ux, train_ux = unit_rows(x), unit_rows(train_x)
                        norm_x = np.log(np.linalg.norm(x, axis=1))
                        norm_dx = upper(np.abs(norm_x[:, None] - norm_x[None, :]))
                        for intervention in CONTROLS:
                            for si, seed in enumerate(SEEDS):
                                y, train_y = ys[intervention][si], train_ys[intervention][si]
                                uy, train_uy = unit_rows(y), unit_rows(train_y)
                                variants = {'raw_cosine': (x, y),
                                            'separate_unit_centers': (ux - train_ux.mean(0), uy - train_uy.mean(0)),
                                            'separate_raw_centers': (x - train_x.mean(0), y - train_y.mean(0)),
                                            'original_unit_center_only': (ux - train_ux.mean(0), uy - train_ux.mean(0)),
                                            'original_raw_center_only': (x - train_x.mean(0), y - train_x.mean(0))}
                                norm_y = np.log(np.linalg.norm(y, axis=1))
                                norm_dy = upper(np.abs(norm_y[:, None] - norm_y[None, :]))
                                controls = [upper(input_rdm), norm_dx, norm_dy]
                                values = {}
                                for name, (a, b) in variants.items():
                                    da, db = rdm(a), rdm(b)
                                    values[name] = {'rdm_rho': rank_correlation(upper(da), upper(db)),
                                                    'input_and_norm_adjusted_rho': partial_rank(upper(da), upper(db), controls)}
                                    if precision == 'fp32' and intervention == 'token_shuffle':
                                        negatives.append({'model': family, 'state': state, 'seed': seed,
                                                          'calibration_population': calibration, 'test_population': test,
                                                          'method': name, **identity_calibration(a, b, permutations)})
                                edx, edy = upper(euclidean_square(x)), upper(euclidean_square(y))
                                values['raw_euclidean'] = {'rdm_rho': rank_correlation(edx, edy),
                                                          'input_and_norm_adjusted_rho': partial_rank(edx, edy, controls),
                                                          'original_norm_distance_rho': rank_correlation(edx, norm_dx),
                                                          'shuffled_norm_distance_rho': rank_correlation(edy, norm_dy)}
                                rows.append({'model': family, 'state': state, 'precision': precision,
                                             'intervention': intervention, 'seed': seed,
                                             'calibration_population': calibration, 'test_population': test,
                                             'methods': values, 'norm_distance_rho': rank_correlation(norm_dx, norm_dy)})
        print(json.dumps({'model': family, 'geometry_rows': len(rows), 'negative_control_rows': len(negatives)}), flush=True)
    result = {'schema_version': 'signal-audit-geometry-refinement-v6',
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'decision': receipt('experiments/signal-audit-v6/iteration-3-decision.json'),
              'code': [receipt(__file__), receipt('src/sonnet_geometry/signal_audit.py')],
              'rows': rows, 'identity_negative_controls': negatives,
              'poem_index_permutations': [p.tolist() for p in permutations],
              'limits': 'All metric variants are post-selection. Separate condition centering changes the estimand. '
                        'Other-author centers use no test-population data. Norm/input projections are in-sample descriptive '
                        'sensitivities, not conditional independence or semantic-information tests. Identity permutations '
                        'calibrate the score, not a literary hypothesis.',
              'new_model_calls': 0, 'elapsed_seconds': time.monotonic() - start}
    write_exclusive(out, canonical_json(result))
    print(json.dumps({'complete': True, 'output': receipt(out), 'seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
