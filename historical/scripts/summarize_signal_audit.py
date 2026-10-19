"""Summarize every v6 diagnostic group and render shareable scientific figures."""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_signal_audit import receipt
from sonnet_geometry.storage import canonical_json, write_exclusive


BASE = Path('outputs/signal-audit-v6')
OUT = Path('experiments/signal-audit-v6')


def describe(values):
    valid = [float(v) for v in values if v is not None and np.isfinite(v)]
    return {'defined': len(valid), 'total': len(values), 'values': values,
            'mean': float(np.mean(valid)) if valid else None, 'median': float(np.median(valid)) if valid else None,
            'minimum': min(valid) if valid else None, 'maximum': max(valid) if valid else None,
            'positive': sum(v > 0 for v in valid), 'negative': sum(v < 0 for v in valid)}


def groups(rows, fields):
    result = defaultdict(list)
    for r in rows:
        ident = tuple(tuple(r[f]) if isinstance(r[f], list) else r[f] for f in fields)
        result[ident].append(r)
    for ident, entries in result.items():
        yield {f: list(v) if isinstance(v, tuple) else v for f, v in zip(fields, ident, strict=True)}, entries


def build(data):
    first, common, structure, calibration, endpoint, geometry, magnitude, components, audit, response_audit = data
    fields = ['population', 'model', 'precision', 'intervention', 'state', 'pooling']
    grid = []
    for pop in first['populations']:
        for identity, rows in groups(pop['paired'], fields):
            r = next(r for r in pop['repeatability'] if all(r[k] == v for k, v in identity.items()))
            grid.append({**identity, 'rdm_rho': describe([r['rdm_rho'] for r in rows]),
                         'identity_top1': describe([r['identity']['top1'] for r in rows]),
                         'centered_identity_top1': describe([r['condition_centered_identity']['top1'] if r['condition_centered_identity'] else None for r in rows]),
                         'biased_cka': describe([r['unit_gram_biased_cka'] for r in rows]),
                         'debiased_hsic_cka': describe([r['unit_gram_unbiased_cka'] for r in rows]),
                         'chance_adjusted_neighbors': describe([r['neighbors']['mean_chance_adjusted_overlap'] for r in rows]),
                         'spread_ratio': describe([r['perturbed_cloud']['unit_spread'] / r['original_cloud']['unit_spread']
                                                   if r['original_cloud']['unit_spread'] > 1e-12 else None for r in rows]),
                         'original_effective_dimension': rows[0]['original_cloud']['effective_dimension'],
                         'repeatability': {k: v for k, v in r.items() if k not in fields and k != 'shuffle_clouds'}})
    transfer = []
    for ident, rows in groups(common['cross_population'], ['model', 'precision', 'state', 'intervention', 'calibration_population', 'test_population']):
        transfer.append({**ident, 'scores': {mode: {metric: describe([r['scores'][mode][metric] for r in rows])
                                                   for metric in ('top1', 'mrr')} for mode in rows[0]['scores']}})
    halves = []
    for ident, rows in groups(common['half_split'], ['population', 'model', 'precision', 'state', 'intervention']):
        halves.append({**ident, 'scores': {mode: describe([float(np.mean([r['scores'][mode]['top1'] for r in rows if r['seed'] == seed]))
                                                        for seed in range(2026101801, 2026101806)]) for mode in rows[0]['scores']},
                       'calibration_size': 12, 'candidate_size': 12, 'dependent_split_count': 10})
    structural = []
    for ident, rows in groups(structure['contrasts'], ['population', 'model', 'precision', 'state', 'pooling']):
        structural.append({**ident, **{metric: {'full': describe([r[metric] for r in rows if r['omitted_poem'] is None]),
                                                'omissions': describe([r[metric] for r in rows if r['omitted_poem'] is not None])}
                                         for metric in ('line_minus_token', 'line_minus_word', 'word_minus_token')}})
    geometric = []
    for ident, rows in groups(geometry['rows'], ['model', 'state', 'precision', 'intervention', 'calibration_population', 'test_population']):
        geometric.append({**ident, 'methods': {mode: {metric: describe([r['methods'][mode][metric] for r in rows])
                                                        for metric in rows[0]['methods'][mode]} for mode in rows[0]['methods']}})
    response = []
    for ident, rows in groups(magnitude['rows'], ['population', 'models', 'precision', 'pooling', 'state', 'intervention']):
        response.append({**ident, 'correlations': {metric: {control: describe([r['correlations'][metric][control] for r in rows])
                                                               for control in rows[0]['correlations'][metric]}
                                                 for metric in rows[0]['correlations']}})
    compact_components = [{k: v for k, v in r.items() if k not in ['within_poem_seed_permutation_reference', 'per_poem_seed_rank_rhos']}
                          | {'per_poem_seed_rank_rho_summary': describe(r['per_poem_seed_rank_rhos'])} for r in components['rows']]
    return {'schema_version': 'signal-audit-complete-summary-v6', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'sources': [receipt(BASE/name) for name in NAMES], 'code': receipt(__file__),
            'status': 'Post-selection exploratory evidence; no new literary confirmation or mechanism',
            'coverage': {'paired_diagnostics': sum(len(p['paired']) for p in first['populations']),
                         'repeatability_cells': len(grid), 'shared_response_cells': sum(len(p['shared_response']) for p in first['populations']),
                         'calibration_half_split_rows': len(common['half_split']), 'cross_author_transfer_rows': len(common['cross_population']),
                         'structural_seed_omission_rows': len(structure['contrasts']), 'structural_depth_interactions': len(structure['depth_interactions']),
                         'endpoint_nuisance_seed_omission_rows': len(endpoint['rows']), 'metric_refinement_rows': len(geometry['rows']),
                         'magnitude_response_rows': len(magnitude['rows']), 'response_components': len(components['rows']),
                         'native_depth_spectra': len(first['native_depth']['spectra'])},
            'diagnostic_grid': grid, 'cross_author_identity': transfer, 'disjoint_calibration_identity': halves,
            'structural_grid': structural, 'geometry_metrics': geometric, 'magnitude_response_grid': response,
            'response_components': compact_components,
            'metric_calibration': [{k: v for k, v in r.items() if k != 'permutation_reference'} for r in calibration['rows']],
            'native_depth': first['native_depth'],
            'independent_audits': [audit, response_audit], 'independent_check_count': audit['checks'] + response_audit['checks'],
            'independent_checks_pass': audit['passes'] and response_audit['passes'], 'new_model_calls': 0,
            'new_conservative_compute_usd': 0,
            'limits': ['All exploration occurs after both population outcomes were known; no fresh held-out author.',
                       'All search cells and unsuccessful refinements retained. No search-adjusted inferential p-values.',
                       'Same-seed, omission, split and permutation ranges are dependent sensitivities, not confidence intervals.',
                       'Identification can rely on retained lexical inventories and is not semantic or literary usefulness.',
                       'Centering does not generally restore the original relational geometry.',
                       'Debiased CKA is a ratio of unbiased HSIC estimates, not an unbiased ratio estimator.',
                       'No human responses or new external labels. No causal author, syntax or circuit claim.']}


def save_figure(fig, name, values, summary_receipt):
    folder = OUT/'figures'; folder.mkdir(exist_ok=True)
    for suffix in ('png', 'pdf'):
        target = folder/f'{name}.{suffix}'
        if target.exists():
            raise FileExistsError(target)
        fig.savefig(target, dpi=180, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    write_exclusive(folder/f'{name}.receipt.json', canonical_json({'summary': summary_receipt,
                    'implementation': receipt(__file__), 'plotted_values': values,
                    'bands': 'Dependent five-seed min/max; not confidence intervals.',
                    'files': [receipt(folder/f'{name}.{s}') for s in ('png', 'pdf')]}))


def figures(summary):
    sr = receipt(OUT/'summary.json'); plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    families, authors = ['qwen', 'gemma', 'muse'], ['sidney', 'spenser']
    colors = ['#68737d', '#1264a3']; values = []
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True, sharex=True)
    labels = []
    for i, (model, test) in enumerate((m, p) for m in families for p in authors):
        ident = next(r for r in summary['cross_author_identity'] if (r['model'], r['precision'], r['state'], r['intervention'], r['test_population'])
                     == (model, 'fp32', 'final_raw', 'token_shuffle', test))
        geometry = next(r for r in summary['geometry_metrics'] if (r['model'], r['precision'], r['state'], r['intervention'], r['test_population'])
                        == (model, 'fp32', 'final_raw', 'token_shuffle', test))
        labels.append(f'{model.title()}\n{test.title()} test')
        row = {'model': model, 'test_population': test, 'calibration_population': ident['calibration_population'],
               'identity': [ident['scores'][k]['top1'] for k in ['raw', 'calibration_centered']],
               'geometry': [geometry['methods'][k]['rdm_rho'] for k in ['raw_cosine', 'separate_unit_centers']]}
        values.append(row)
        for ax, metric in zip(axes, ['identity', 'geometry'], strict=True):
            for j, stats in enumerate(row[metric]):
                position = i + (j - .5) * .34
                ax.bar(position, stats['median'], width=.31, color=colors[j], label=['Raw', 'Other-author calibration center'][j] if i == 0 else None)
                ax.errorbar(position, stats['median'], yerr=[[stats['median'] - stats['minimum']], [stats['maximum'] - stats['median']]],
                            fmt='none', ecolor='black', capsize=3, lw=1)
    axes[0].axhline(1 / 24, ls=':', color='black', lw=1, label='Identity chance: 1/24')
    axes[0].set_ylabel('Correct poem identity (fraction)'); axes[0].set_ylim(0, 1.05); axes[0].legend(loc='upper right', fontsize=9)
    axes[0].set_title('Identity becomes recoverable; the original relationships do not return')
    axes[1].set_ylabel('Original/shuffle RDM Spearman ρ'); axes[1].set_ylim(-.16, 1.0); axes[1].axhline(0, lw=.7, color='black')
    axes[1].set_xticks(range(6), labels)
    fig.suptitle('Final raw mean, exact-token shuffle, FP32 — both transfer directions', fontsize=14)
    fig.supxlabel('Centers use only the other author. Bars: five-seed medians; whiskers: dependent ranges, not confidence intervals.', fontsize=9)
    save_figure(fig, 'identity-and-relations', values, sr)
    states = ['early', 'depth75', 'final_raw', 'final_normalized']; labels = ['Early', 'Depth .75', 'Final raw', 'Final norm.']
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.7), constrained_layout=True, sharex=True, sharey=True); values = []
    for i, pop in enumerate(authors):
        for j, model in enumerate(families):
            ax = axes[i, j]
            for control, color, label in [('line_shuffle', '#1264a3', 'Move whole lines'), ('token_shuffle', '#c75328', 'Shuffle tokens within lines'),
                                          ('word_shuffle', '#68737d', 'Shuffle words (retokenized)')]:
                rows = [next(r for r in summary['diagnostic_grid'] if (r['population'], r['model'], r['precision'], r['intervention'], r['state'], r['pooling'])
                             == (pop, model, 'fp32', control, state, 'mean')) for state in states]
                med = [r['rdm_rho']['median'] for r in rows]
                ax.plot(range(4), med, marker='o', ms=4, color=color, label=label)
                ax.fill_between(range(4), [r['rdm_rho']['minimum'] for r in rows], [r['rdm_rho']['maximum'] for r in rows], color=color, alpha=.12)
                values.append({'population': pop, 'model': model, 'intervention': control, 'states': states, 'rho': [r['rdm_rho'] for r in rows]})
            ax.axhline(0, lw=.5, color='black'); ax.set_ylim(-.32, 1.02); ax.set_xticks(range(4), labels, rotation=15)
            ax.set_title(f'{pop.title()} / {model.title()}')
            if j == 0:
                ax.set_ylabel('Original/perturbed RDM ρ')
    axes[0, 0].legend(fontsize=7.3, loc='lower left')
    fig.suptitle('The unit of disruption matters across depth — FP32 mean readout', fontsize=14)
    fig.supxlabel('Sampled states are categorical; final normalization is separate. Bands: dependent five-seed ranges. Word shuffles change token inventories.', fontsize=9)
    save_figure(fig, 'local-and-global-order', values, sr)
    pairs = [['qwen', 'gemma'], ['qwen', 'muse'], ['gemma', 'muse']]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharey='row'); values = []
    for col, pop in enumerate(authors):
        for i, pair in enumerate(pairs):
            r = next(r for r in summary['response_components'] if r['models'] == pair and
                     (r['population'], r['precision'], r['intervention'], r['state'], r['pooling']) == (pop, 'fp32', 'word_shuffle', 'early', 'last_lexical'))
            for j, (metric, color, label) in enumerate([('same_seed', '#1264a3', 'Same scramble'), ('different_seed', '#c75328', 'Different scramble')]):
                axes[0, col].bar(i + (j - .5) * .26, r[metric]['median'], width=.24, color=color, label=label if i == 0 else None)
            axes[0, col].scatter(i, r['mean_across_seeds_poem_response_rho'], marker='D', color='black', s=28, label='Five-seed poem average' if i == 0 else None)
            ref = r['within_poem_seed_permutation_summary']
            axes[1, col].vlines(i, ref['minimum'], ref['maximum'], color='#b8bec4', lw=14, label='199 seed-label permutation range' if i == 0 else None)
            axes[1, col].scatter(i, r['within_poem_seed_rank_correlation'], color='#1264a3', s=65, zorder=3, label='Actual within-poem response ranks' if i == 0 else None)
            values.append(r)
        axes[0, col].set_title(f'{pop.title()}: which poems change more?'); axes[1, col].set_title(f'{pop.title()}: which scramble changes a given poem more?')
        for ax in axes[:, col]:
            ax.set_xticks(range(3), ['Qwen–Gemma', 'Qwen–Muse', 'Gemma–Muse']); ax.axhline(0, color='black', lw=.6)
        axes[0, col].set_ylim(0, 1); axes[1, col].set_ylim(-.4, 1)
    axes[0, 0].set_ylabel('Across-poem response rank correlation'); axes[1, 0].set_ylabel('Within-poem seed-rank correlation')
    axes[0, 0].legend(fontsize=8); axes[1, 0].legend(fontsize=8, loc='upper left')
    fig.suptitle('Shared early lexical-endpoint sensitivity has two components', fontsize=14)
    fig.supxlabel('Post-selected word-shuffle readout, FP32. Same surface text across families; retokenization remains. Permutation ranges are descriptive, not intervals.', fontsize=9)
    save_figure(fig, 'shared-response-components', values, sr)


NAMES = ['first-pass.json', 'iteration-2-common.json', 'iteration-2-structure.json', 'iteration-2-calibration.json',
         'iteration-2-endpoint.json', 'iteration-3-geometry.json', 'iteration-4-responses.json',
         'iteration-5-response-components.json', 'independent-numerical-audit.json', 'independent-response-audit.json']


def main():
    if (OUT/'summary.json').exists():
        raise FileExistsError(OUT/'summary.json')
    data = [json.loads((BASE/name).read_text()) for name in NAMES]
    summary = build(data)
    assert summary['independent_checks_pass'] is True
    write_exclusive(OUT/'summary.json', canonical_json(summary))
    figures(summary)
    print(json.dumps({'summary': receipt(OUT/'summary.json'), 'coverage': summary['coverage'], 'checks': summary['independent_check_count']}))


if __name__ == '__main__':
    main()
