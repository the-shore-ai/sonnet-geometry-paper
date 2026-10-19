"""Targeted, explicitly post-selection checks of v6 first-pass leads.

The iteration decision is saved before these calculations. Outputs are
create-only. No inference, external writes, or statistical p-values.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
from safetensors import safe_open

from analyze_signal_audit import POPULATIONS, CONTROLS, get_matrix, receipt, finite_summary
from sonnet_geometry.confirmation_analysis import condition_name, SEEDS
from sonnet_geometry.signal_audit import (
    cka_from_grams, partial_rank, rank_correlation, retrieval, unit_rows, upper,
)
from sonnet_geometry.storage import canonical_json, write_exclusive
from sonnet_geometry.text_control_analysis import FAMILIES, STATES, POOLS


def compact_retrieval(a, b):
    if min(np.min(np.linalg.norm(x, axis=1)) for x in (a, b)) <= 1e-12:
        return None
    return {k: v for k, v in retrieval(a, b).items() if k != 'directions'}


def calibration_modes(xtrain, ytrain, xtest, ytest, other_seed_train):
    xmean, ymean = xtrain.mean(0), ytrain.mean(0)
    return {'raw': compact_retrieval(xtest, ytest),
            'calibration_centered': compact_retrieval(xtest - xmean, ytest - ymean),
            'calibration_translated': compact_retrieval(xtest, ytest - (ymean - xmean)),
            'other_seed_calibration_centered': compact_retrieval(xtest - xmean, ytest - other_seed_train.mean(0))}


def common_shift(first):
    studies = {p: json.loads(Path(path).read_text()) for p, path in POPULATIONS.items()}
    rows, transfer, shifts = [], [], []
    split_seeds = list(range(2026101901, 2026101911))
    for family in FAMILIES:
        with ExitStack() as stack:
            handles = {}
            for pop, study in studies.items():
                files = study['models'][family]['source_files']
                pooled = next(x for x in files if x['path'].endswith('.safetensors'))
                assert receipt(pooled['path'])['sha256'] == pooled['sha256']
                handles[pop] = stack.enter_context(safe_open(pooled['path'], framework='numpy'))
            for precision in ('fp32', 'bf16'):
                for state in STATES[1:]:
                    matrices = {}
                    for pop, study in studies.items():
                        h, ids = handles[pop], study['poems']
                        original = unit_rows(get_matrix(h, ids, condition_name('original', None, precision), state, 'mean'))
                        perturbed = {c: [unit_rows(get_matrix(h, ids, condition_name(c, s, precision), state, 'mean'))
                                          for s in SEEDS] for c in CONTROLS}
                        matrices[pop] = original, perturbed
                    for intervention in CONTROLS:
                        for seed_index, seed in enumerate(SEEDS):
                            ident = {'model': family, 'precision': precision, 'state': state, 'pooling': 'mean',
                                     'intervention': intervention, 'seed': seed}
                            for pop, study in studies.items():
                                original, all_changed = matrices[pop]
                                changed = all_changed[intervention][seed_index]
                                other = all_changed[intervention][(seed_index + 1) % len(SEEDS)]
                                ids = study['poems']
                                for split_seed in split_seeds:
                                    order = np.random.default_rng(split_seed).permutation(len(ids))
                                    train, test = order[:12], order[12:]
                                    rows.append({**ident, 'population': pop, 'split_seed': split_seed,
                                                 'calibration_poems': [ids[i] for i in train],
                                                 'test_poems': [ids[i] for i in test],
                                                 'calibration_other_seed': SEEDS[(seed_index + 1) % len(SEEDS)],
                                                 'scores': calibration_modes(original[train], changed[train],
                                                     original[test], changed[test], other[train])})
                            a, b = [matrices[p] for p in POPULATIONS]
                            da = a[1][intervention][seed_index].mean(0) - a[0].mean(0)
                            db = b[1][intervention][seed_index].mean(0) - b[0].mean(0)
                            denominator = np.linalg.norm(da) * np.linalg.norm(db)
                            shifts.append({**ident, 'cross_population_shift_cosine': float(da @ db / denominator)
                                           if denominator > 1e-20 else None,
                                           'cross_population_shift_norm_ratio': float(np.linalg.norm(db) / np.linalg.norm(da))
                                           if np.linalg.norm(da) > 1e-12 else None})
                            for calibration, test_pop in [('sidney', 'spenser'), ('spenser', 'sidney')]:
                                xtrain, ytrain = matrices[calibration]
                                xtest, ytest = matrices[test_pop]
                                transfer.append({**ident, 'calibration_population': calibration, 'test_population': test_pop,
                                                 'calibration_poems': studies[calibration]['poems'],
                                                 'test_poems': studies[test_pop]['poems'],
                                                 'scores': calibration_modes(xtrain, ytrain[intervention][seed_index], xtest,
                                                     ytest[intervention][seed_index],
                                                     ytrain[intervention][(seed_index + 1) % len(SEEDS)])})
        print(json.dumps({'part': 'common_shift', 'model': family, 'half_split_rows': len(rows),
                          'cross_population_rows': len(transfer)}), flush=True)
    return {'half_split': rows, 'cross_population': transfer, 'common_shift_directions': shifts,
            'limits': 'Calibration uses only other poems. Half splits have 12 candidates, cross-author transfer 24. '
                      'Ten splits and five seeds are dependent descriptive checks. This recovers inventory-preserving identity, '
                      'not literary meaning. Centering is a post-selection measurement and changes the estimand.'}


def structure(first):
    rows, interactions = [], []
    for pop, path in POPULATIONS.items():
        analysis = json.loads(Path(path).read_text())
        ids = analysis['poems']
        tri = np.triu_indices(len(ids), 1)
        distances = {}
        for r in analysis['rdms']:
            d = np.zeros((len(ids), len(ids)))
            d[tri] = r['upper_triangle']
            d += d.T
            distances[r['model'], r['condition'], r['state'], r['pooling']] = d
        for family in FAMILIES:
            for precision in ('fp32', 'bf16'):
                for state in STATES:
                    for pool in POOLS:
                        for seed in SEEDS:
                            condition = lambda c: condition_name(c, None if c == 'original' else seed, precision)
                            original = distances[family, condition('original'), state, pool]
                            ds = {c: distances[family, condition(c), state, pool] for c in CONTROLS}
                            for omitted in [None, *ids]:
                                ix = np.asarray([i for i, n in enumerate(ids) if n != omitted])
                                a = upper(original[np.ix_(ix, ix)])
                                values = {c: rank_correlation(a, upper(d[np.ix_(ix, ix)])) for c, d in ds.items()}
                                delta = lambda a, b: None if values[a] is None or values[b] is None else values[a] - values[b]
                                rows.append({'population': pop, 'model': family, 'precision': precision, 'state': state,
                                             'pooling': pool, 'seed': seed, 'omitted_poem': omitted,
                                             'stabilities': values, 'line_minus_token': delta('line_shuffle', 'token_shuffle'),
                                             'line_minus_word': delta('line_shuffle', 'word_shuffle'),
                                             'word_minus_token': delta('word_shuffle', 'token_shuffle')})
            print(json.dumps({'part': 'structure', 'population': pop, 'model': family, 'rows': len(rows)}), flush=True)
        index = {(r['model'], r['precision'], r['state'], r['pooling'], r['seed'], r['omitted_poem']): r for r in rows if r['population'] == pop}
        for (family, precision, state, pool, seed, omitted), r in index.items():
            if state != 'depth75':
                continue
            early = index[family, precision, 'early', pool, seed, omitted]
            a, b = r['line_minus_token'], early['line_minus_token']
            interactions.append({'population': pop, 'model': family, 'precision': precision, 'pooling': pool,
                                 'seed': seed, 'omitted_poem': omitted,
                                 'depth75_minus_early_line_token_interaction': None if a is None or b is None else a - b})
    return {'contrasts': rows, 'depth_interactions': interactions,
            'limits': 'All omission choices retained. Line and token controls preserve inventories but differ in disruption '
                      'unit, fixed punctuation/endpoints and preserved couplet; neither contrast identifies syntax causally. '
                      'Word controls retokenize and are not inventory matched.'}


def calibration(first):
    rows = []
    rng = np.random.default_rng(2026101920)
    permutations = [rng.permutation(24) for _ in range(199)]
    for pop, path in POPULATIONS.items():
        analysis = json.loads(Path(path).read_text())
        tri = np.triu_indices(24, 1)
        distances = {}
        for r in analysis['rdms']:
            if r['pooling'] != 'mean' or 'fp32' not in r['condition']:
                continue
            d = np.zeros((24, 24)); d[tri] = r['upper_triangle']; d += d.T
            distances[r['model'], r['condition'], r['state']] = d
        for family in FAMILIES:
            for state in STATES[1:]:
                original = distances[family, 'original_padded_fp32', state]
                for seed in SEEDS:
                    changed = distances[family, condition_name('token_shuffle', seed, 'fp32'), state]
                    actual = cka_from_grams(1 - original, 1 - changed)
                    reference = [cka_from_grams(1 - original, 1 - changed[np.ix_(p, p)]) for p in permutations]
                    rows.append({'population': pop, 'model': family, 'state': state, 'seed': seed,
                                 'comparison': 'original_vs_token_shuffle', 'observed': actual,
                                 'permutation_reference': reference,
                                 'permutation_medians': {k: finite_summary([r[k] for r in reference]) for k in actual}})
        for left, right in combinations(FAMILIES, 2):
            for state in STATES[1:]:
                a, b = [distances[f, 'original_padded_fp32', state] for f in (left, right)]
                actual = cka_from_grams(1 - a, 1 - b)
                reference = [cka_from_grams(1 - a, 1 - b[np.ix_(p, p)]) for p in permutations]
                rows.append({'population': pop, 'models': [left, right], 'state': state,
                             'comparison': 'original_cross_model', 'observed': actual,
                             'permutation_reference': reference,
                             'permutation_medians': {k: finite_summary([r[k] for r in reference]) for k in actual}})
    return {'rows': rows, 'permutations': [p.tolist() for p in permutations],
            'limits': 'Poem index permutations are a descriptive metric calibration, not a valid literary-content null '
                      'or adjusted significance test. The field named unbiased_cka is the normalized ratio of unbiased '
                      'HSIC estimates, conventionally called debiased CKA; the ratio itself is not unbiased.'}


def endpoint(first):
    rows, sources, details = [], [], []
    for pop, path in POPULATIONS.items():
        study = json.loads(Path(path).read_text()); ids = study['poems']; features = {}
        for family in FAMILIES:
            report = next(x for x in study['models'][family]['source_files'] if x['path'].endswith('report.json'))
            assert receipt(report['path'])['sha256'] == report['sha256']
            sources.append(report)
            wrapper = json.loads(Path(report['path']).read_text()); actual = wrapper.get('report', wrapper)
            cases = {(c['sonnet'], c['condition']): c for c in actual['cases']}
            for intervention in CONTROLS:
                for seed in SEEDS:
                    data = []
                    for n in ids:
                        a, b = [cases[n, condition_name(c, None if c == 'original' else seed, 'fp32')]['tokenization']
                                for c in ('original', intervention)]
                        ia, ib = a['input_ids'], b['input_ids']
                        ea, eb = a['lexical_indices'][-1], b['lexical_indices'][-1]
                        offsets = a['offsets']; line_start = a['text'].rfind('\n') + 1
                        last_line_indices = [i for i, (start, end) in enumerate(offsets) if end > line_start and end > start]
                        dose = lambda indices: float(np.mean([i >= len(ib) or ia[i] != ib[i] for i in indices])) if len(indices) else 0.
                        values = {'token_count': len(ia), 'token_count_change': len(ib) - len(ia),
                                  'abs_token_count_change': abs(len(ib) - len(ia)),
                                  'endpoint_position_change': eb - ea, 'abs_endpoint_position_change': abs(eb - ea),
                                  'endpoint_token_changes': int(ia[ea] != ib[eb]),
                                  'last_line_tokens': len(last_line_indices), 'last_line_changed_fraction': dose(last_line_indices),
                                  'last8_changed_fraction': dose(range(max(0, ea - 8), ea)),
                                  'last16_changed_fraction': dose(range(max(0, ea - 16), ea)),
                                  'last_line_words': len(a['text'].splitlines()[-1].split()),
                                  'last_lexical_token_id': ia[ea]}
                        data.append(values)
                        details.append({'population': pop, 'model': family, 'intervention': intervention,
                                        'seed': seed, 'poem': n, **values})
                    features[family, intervention, seed] = data
            del wrapper, actual, cases
        first_pop = next(p for p in first['populations'] if p['population'] == pop)
        paired = {(r['model'], r['precision'], r['intervention'], r['seed'], r['state'], r['pooling']): r for r in first_pop['paired']}
        for left, right in combinations(FAMILIES, 2):
            for precision in ('fp32', 'bf16'):
                for intervention in CONTROLS:
                    for seed in SEEDS:
                        for state in ('early', 'depth75', 'final_raw', 'final_normalized'):
                            for pool in ('last_lexical', 'mean'):
                                a, b = [paired[f, precision, intervention, seed, state, pool] for f in (left, right)]
                                fa, fb = [features[f, intervention, seed] for f in (left, right)]
                                base = [[v['token_count'] for v in fs] for fs in (fa, fb)]
                                # A small prespecified progression separates length and local-dose sensitivities.
                                controls = {'length': base,
                                            'length_and_position_change': [*base, *[[v[k] for v in fs] for fs in (fa, fb)
                                                for k in ('abs_endpoint_position_change', 'token_count_change')]],
                                            'length_and_local_disruption': [*base, *[[v[k] for v in fs] for fs in (fa, fb)
                                                for k in ('last_line_tokens', 'last8_changed_fraction', 'last16_changed_fraction')]]}
                                effects = a['per_poem_cosine_displacement'], b['per_poem_cosine_displacement']
                                for omitted in [None, *ids]:
                                    indices = [i for i, n in enumerate(ids) if n != omitted]
                                    take = lambda x: np.asarray(x)[indices]
                                    result = {k: partial_rank(*[take(x) for x in effects], [take(x) for x in cs]) for k, cs in controls.items()}
                                    rows.append({'population': pop, 'models': [left, right], 'precision': precision,
                                                 'intervention': intervention, 'seed': seed, 'state': state, 'pooling': pool,
                                                 'omitted_poem': omitted, 'raw_rho': rank_correlation(*[take(x) for x in effects]),
                                                 'adjusted': result})
        print(json.dumps({'part': 'endpoint', 'population': pop, 'rows': len(rows)}), flush=True)
    return {'rows': rows, 'input_features': details, 'sources': sources,
            'limits': 'Local nuisance controls were chosen after seeing high early lexical-endpoint response agreement. '
                      '24 poems and several correlated covariates permit fragile residual fits. No inferential p-values. '
                      'Positionwise token dose is less interpretable for retokenized word controls. Last-token IDs are '
                      'categorical and are not treated as numeric covariates.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part', choices=['common', 'structure', 'calibration', 'endpoint'])
    args = parser.parse_args()
    out = Path(f'outputs/signal-audit-v6/iteration-2-{args.part}.json')
    if out.exists():
        raise FileExistsError(out)
    start = time.monotonic()
    source = Path('outputs/signal-audit-v6/first-pass.json')
    first = json.loads(source.read_text())
    functions = {'common': common_shift, 'structure': structure, 'calibration': calibration, 'endpoint': endpoint}
    result = {'schema_version': 'signal-audit-targeted-checks-v6', 'part': args.part,
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'decision': receipt('experiments/signal-audit-v6/iteration-2-decision.json'),
              'first_pass': receipt(source), 'code': [receipt(__file__), receipt('src/sonnet_geometry/signal_audit.py')],
              **functions[args.part](first), 'new_model_calls': 0}
    result['elapsed_seconds'] = time.monotonic() - start
    write_exclusive(out, canonical_json(result))
    print(json.dumps({'complete': True, 'part': args.part, 'output': receipt(out), 'seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
