"""Complete exploratory diagnostic grid over saved Sidney/Spenser vectors.

No inference, cloud mutation, model loading, outcome selection, or p-values.
Run from the repository root. Detailed outputs are private derived artifacts.
"""
import argparse
from datetime import datetime, timezone
import gc
from itertools import combinations
import json
from pathlib import Path
import platform
import time

import numpy as np
from safetensors import safe_open
import scipy

from sonnet_geometry.confirmation_analysis import condition_name, SEEDS
from sonnet_geometry.depth_discovery import load_model_rdms
from sonnet_geometry.signal_audit import (
    cka_from_grams, cloud_summary, neighbors, paired_metrics, partial_rank,
    rank_correlation, rdm, seed_repeatability, spectrum_from_gram, upper,
)
from sonnet_geometry.storage import canonical_json, sha256_file, write_exclusive
from sonnet_geometry.text_control_analysis import FAMILIES, STATES, POOLS, key


POPULATIONS = {
    'sidney': 'outputs/confirmation-v4/muse-finalization-repair/strict-analysis.json',
    'spenser': 'outputs/confirmation-v5-spenser/strict-analysis.json',
}
CONTROLS = ('word_shuffle', 'line_shuffle', 'token_shuffle')


def receipt(path):
    p = Path(path)
    return {'path': str(p), 'sha256': sha256_file(p), 'bytes': p.stat().st_size}


def finite_summary(values):
    valid = [x for x in values if x is not None and np.isfinite(x)]
    return {'defined': len(valid), 'total': len(values), 'median': float(np.median(valid)) if valid else None,
            'minimum': float(min(valid)) if valid else None, 'maximum': float(max(valid)) if valid else None}


def get_matrix(handle, numbers, condition, state, pool):
    return np.stack([handle.get_tensor(key(n, condition, state, pool)) for n in numbers]).astype(np.float64)


def metadata(case):
    token = case['tokenization']
    ids = token['input_ids']
    text = token['text']
    return {'token_count': len(ids), 'character_count': len(text),
            'word_count': len(text.split()), 'text_sha256': token['text_sha256'], 'input_ids': ids}


def token_change(original, changed):
    a, b = original['input_ids'], changed['input_ids']
    common = min(len(a), len(b))
    return (sum(a[i] != b[i] for i in range(common)) + abs(len(a) - len(b))) / max(len(a), len(b))


def population_grid(population, analysis_path):
    analysis = json.loads(Path(analysis_path).read_text())
    assert analysis['schema_version'] == 'depth-confirmation-analysis-v4'
    numbers = analysis['poems']
    all_rows, repeatability, sources = [], [], [receipt(analysis_path)]
    nuisance, responses, original_rdms, original_meta = {}, {}, {}, {}
    for family in FAMILIES:
        files = analysis['models'][family]['source_files']
        for entry in files:
            actual = receipt(entry['path'])
            assert actual['sha256'] == entry['sha256'] and actual['bytes'] == entry['bytes']
            sources.append(actual)
        report_path = next(x['path'] for x in files if x['path'].endswith('report.json'))
        pooled_path = next(x['path'] for x in files if x['path'].endswith('.safetensors'))
        wrapper = json.loads(Path(report_path).read_text())
        report = wrapper.get('report', wrapper)
        assert report['complete'] is True and len(report['cases']) == 768
        cases = {(c['sonnet'], c['condition']): metadata(c) for c in report['cases']}
        del wrapper, report
        original_meta[family] = [cases[n, 'original_padded_fp32'] for n in numbers]
        lengths = np.asarray([x['token_count'] for x in original_meta[family]])
        length_diff = upper(np.abs(lengths[:, None] - lengths[None, :]))
        length_mean = upper((lengths[:, None] + lengths[None, :]) / 2)
        with safe_open(pooled_path, framework='numpy') as handle:
            for precision in ('bf16', 'fp32'):
                base_name = condition_name('original', None, precision)
                input_rdm = rdm(get_matrix(handle, numbers, base_name, 'input', 'mean'))
                for state in STATES:
                    for pool in POOLS:
                        original = get_matrix(handle, numbers, base_name, state, pool)
                        original_rdms[family, precision, state, pool] = rdm(original)
                        for intervention in CONTROLS:
                            perturbations = []
                            for seed in SEEDS:
                                name = condition_name(intervention, seed, precision)
                                changed = get_matrix(handle, numbers, name, state, pool)
                                perturbations.append(changed)
                                metrics = paired_metrics(original, changed)
                                changed_input = input_rdm if intervention != 'word_shuffle' else rdm(
                                    get_matrix(handle, numbers, name, 'input', 'mean'))
                                controls = [upper(input_rdm), upper(changed_input), length_diff, length_mean]
                                metrics['input_and_length_adjusted_rdm_correlation'] = partial_rank(
                                    upper(rdm(original)), upper(rdm(changed)), controls)
                                changed_meta = [cases[n, name] for n in numbers]
                                dose = [token_change(a, b) for a, b in zip(original_meta[family], changed_meta, strict=True)]
                                identity = {'population': population, 'model': family, 'precision': precision,
                                            'intervention': intervention, 'seed': seed, 'state': state, 'pooling': pool}
                                row = {**identity, **metrics, 'per_poem_changed_token_fraction': dose,
                                       'per_poem_token_count_change': [b['token_count'] - a['token_count']
                                           for a, b in zip(original_meta[family], changed_meta, strict=True)]}
                                all_rows.append(row)
                                responses[family, precision, intervention, seed, state, pool] = {
                                    'effect': metrics['per_poem_cosine_displacement'], 'dose': dose,
                                    'norm_response': metrics['per_poem_log_norm_ratio'],
                                    'surface_text_hashes': [m['text_sha256'] for m in changed_meta]}
                            repetition = seed_repeatability(original, perturbations)
                            repeated = [row for row in all_rows[-5:]]
                            repetition.update({**{k: v for k, v in identity.items() if k != 'seed'},
                                               'original_to_shuffle_rho': finite_summary([r['rdm_rho'] for r in repeated]),
                                               'interseed_rho_summary': finite_summary(repetition['interseed_rdm_rhos']),
                                               'shuffle_clouds': [cloud_summary(x) for x in perturbations]})
                            repeatability.append(repetition)
                        nuisance[family, precision, state, pool] = {
                            'input': upper(input_rdm), 'length_diff': length_diff, 'length_mean': length_mean}
        del cases
        gc.collect()
        print(json.dumps({'population': population, 'model': family, 'paired_rows': len(all_rows),
                          'repeatability_rows': len(repeatability)}), flush=True)
    shared, cross = [], []
    for left, right in combinations(FAMILIES, 2):
        length_controls = [np.asarray([x['token_count'] for x in original_meta[f]]) for f in (left, right)]
        length_controls.append(np.asarray([x['character_count'] for x in original_meta[left]]))
        for precision in ('bf16', 'fp32'):
            for state in STATES:
                for pool in POOLS:
                    a, b = [original_rdms[f, precision, state, pool] for f in (left, right)]
                    nleft, nright = [nuisance[f, precision, state, pool] for f in (left, right)]
                    controls = [nleft['input'], nright['input'], nleft['length_diff'], nright['length_diff'],
                                nleft['length_mean'], nright['length_mean']]
                    cross.append({'population': population, 'models': [left, right], 'precision': precision,
                                  'state': state, 'pooling': pool, 'original_rho': rank_correlation(upper(a), upper(b)),
                                  'input_and_length_adjusted_rho': partial_rank(upper(a), upper(b), controls),
                                  **cka_from_grams(1 - a, 1 - b)})
                    for intervention in CONTROLS:
                        for seed in SEEDS:
                            ra, rb = [responses[f, precision, intervention, seed, state, pool] for f in (left, right)]
                            effects = [ra['effect'], rb['effect']]
                            shared.append({'population': population, 'models': [left, right], 'precision': precision,
                                           'state': state, 'pooling': pool, 'intervention': intervention, 'seed': seed,
                                           'response_rho': rank_correlation(*effects),
                                           'length_adjusted_response_rho': partial_rank(*effects, length_controls),
                                           'length_and_dose_adjusted_response_rho': partial_rank(
                                               *effects, [*length_controls, ra['dose'], rb['dose']]),
                                           'norm_response_rho': rank_correlation(ra['norm_response'], rb['norm_response']),
                                           'surface_text_hashes_match': ra['surface_text_hashes'] == rb['surface_text_hashes']
                                               if intervention != 'token_shuffle' else None})
    return {'population': population, 'poems': numbers, 'sources': sources,
            'paired': all_rows, 'repeatability': repeatability, 'shared_response': shared, 'original_cross': cross}


def native_depth():
    rows, comparisons, sources = [], [], []
    for family in FAMILIES:
        model = load_model_rdms(f'outputs/full-{family}-v1/report.json')
        sources.append(model.source)
        for pool in model.distances:
            for layer, distance in zip(model.layers, model.distances[pool], strict=True):
                rows.append({'model': family, 'pooling': pool, 'state': layer['tensor'],
                             'normalized_depth': layer['normalized_depth'], **spectrum_from_gram(1 - distance),
                             'mean_pair_distance': float(np.mean(upper(distance))),
                             'input_rho': rank_correlation(upper(distance), upper(model.distances[pool][0]))})
            raw, norm = model.distances[pool][-1], model.final_normalized[pool]
            rows.append({'model': family, 'pooling': pool, 'state': 'final_normalized',
                         'normalized_depth': None, **spectrum_from_gram(1 - norm),
                         'mean_pair_distance': float(np.mean(upper(norm))),
                         'input_rho': rank_correlation(upper(norm), upper(model.distances[pool][0]))})
            comparisons.append({'model': family, 'pooling': pool,
                                'raw_to_normalized_rho': rank_correlation(upper(raw), upper(norm)),
                                'raw_to_normalized_neighbors': neighbors(raw, norm),
                                **cka_from_grams(1 - raw, 1 - norm)})
    return {'scope': 'All 154 original BF16 poems, all raw residual states, both old readouts; original numerical limitations retained.',
            'sources': sources, 'spectra': rows, 'normalization': comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/signal-audit-v6/first-pass.json'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    start = time.monotonic()
    plan = receipt('experiments/signal-audit-v6/plan.json')
    result = {'schema_version': 'exploratory-signal-audit-v6', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'plan': plan, 'code': [receipt(__file__), receipt('src/sonnet_geometry/signal_audit.py')],
              'populations': [population_grid(name, path) for name, path in POPULATIONS.items()],
              'native_depth': native_depth(), 'environment': {'python': platform.python_version(),
                  'numpy': np.__version__, 'scipy': scipy.__version__}, 'new_model_calls': 0,
              'inference': 'Post-outcome descriptive field search; no fresh held-out population, p-values or literary usefulness claims.'}
    result['elapsed_seconds'] = time.monotonic() - start
    write_exclusive(args.output, canonical_json(result))
    print(json.dumps({'complete': True, 'output': receipt(args.output), 'seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
