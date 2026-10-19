"""Iteration 4: challenge shared angular responses with magnitude controls."""
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import re
import time

import numpy as np
from safetensors import safe_open

from analyze_signal_audit import POPULATIONS, CONTROLS, get_matrix, receipt
from sonnet_geometry.confirmation_analysis import condition_name, SEEDS
from sonnet_geometry.signal_audit import unit_rows, rank_correlation, partial_rank
from sonnet_geometry.storage import canonical_json, write_exclusive
from sonnet_geometry.text_control_analysis import FAMILIES, STATES


def main():
    out = Path('outputs/signal-audit-v6/iteration-4-responses.json')
    if out.exists():
        raise FileExistsError(out)
    start = time.monotonic()
    sources, details, rows = [], [], []
    for pop, path in POPULATIONS.items():
        study = json.loads(Path(path).read_text()); ids = study['poems']; features = {}; meta = {}
        for family in FAMILIES:
            files = study['models'][family]['source_files']
            report_entry = next(x for x in files if x['path'].endswith('report.json'))
            pooled_entry = next(x for x in files if x['path'].endswith('.safetensors'))
            for item in [report_entry, pooled_entry]:
                assert receipt(item['path'])['sha256'] == item['sha256']; sources.append(item)
            wrapper = json.loads(Path(report_entry['path']).read_text()); report = wrapper.get('report', wrapper)
            originals = {c['sonnet']: c['tokenization'] for c in report['cases'] if c['condition'] == 'original_padded_fp32'}
            meta[family] = {'length': [len(originals[n]['input_ids']) for n in ids],
                            'last_word_length': [len(re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)*", originals[n]['text'])[-1]) for n in ids],
                            'last_lexical_token_id': [originals[n]['input_ids'][originals[n]['lexical_indices'][-1]] for n in ids]}
            del wrapper, report, originals
            with safe_open(pooled_entry['path'], framework='numpy') as h:
                for precision in ('fp32', 'bf16'):
                    base = condition_name('original', None, precision)
                    for pool in ('mean', 'last_lexical'):
                        input_norm = np.log(np.linalg.norm(get_matrix(h, ids, base, 'input', pool), axis=1))
                        for state in STATES[1:]:
                            x = get_matrix(h, ids, base, state, pool)
                            nx = np.linalg.norm(x, axis=1); ux = unit_rows(x)
                            for intervention in CONTROLS:
                                for seed in SEEDS:
                                    y = get_matrix(h, ids, condition_name(intervention, seed, precision), state, pool)
                                    values = {'cosine': np.clip(1 - np.sum(ux * unit_rows(y), axis=1), 0, 2).tolist(),
                                              'squared_l2': (np.sum((y - x) ** 2, axis=1) / np.median(nx) ** 2).tolist(),
                                              'log_norm_ratio': np.log(np.linalg.norm(y, axis=1) / nx).tolist(),
                                              'original_input_log_norm': input_norm.tolist(), 'original_state_log_norm': np.log(nx).tolist()}
                                    features[family, precision, pool, state, intervention, seed] = values
                                    details.append({'population': pop, 'model': family, 'precision': precision,
                                                    'pooling': pool, 'state': state, 'intervention': intervention,
                                                    'seed': seed, 'poems': ids, **values})
        for left, right in combinations(FAMILIES, 2):
            for precision in ('fp32', 'bf16'):
                for pool in ('mean', 'last_lexical'):
                    for state in STATES[1:]:
                        for intervention in CONTROLS:
                            for seed in SEEDS:
                                a, b = [features[f, precision, pool, state, intervention, seed] for f in (left, right)]
                                controls = {'input_norms': [a['original_input_log_norm'], b['original_input_log_norm']],
                                            'state_norms': [a['original_state_log_norm'], b['original_state_log_norm']],
                                            'input_and_state_norms_and_length': [a['original_input_log_norm'], b['original_input_log_norm'],
                                                a['original_state_log_norm'], b['original_state_log_norm'], meta[left]['length'], meta[right]['length'],
                                                meta[left]['last_word_length']]}
                                correlations = {}
                                for metric in ('cosine', 'squared_l2', 'log_norm_ratio'):
                                    correlations[metric] = {'raw': rank_correlation(a[metric], b[metric]),
                                                           **{k: partial_rank(a[metric], b[metric], cs) for k, cs in controls.items()}}
                                rows.append({'population': pop, 'models': [left, right], 'precision': precision,
                                             'pooling': pool, 'state': state, 'intervention': intervention, 'seed': seed,
                                             'correlations': correlations,
                                             'input_norm_cross_model_rho': rank_correlation(a['original_input_log_norm'], b['original_input_log_norm']),
                                             'state_norm_cross_model_rho': rank_correlation(a['original_state_log_norm'], b['original_state_log_norm']),
                                             'cosine_response_input_norm_rho': [rank_correlation(v['cosine'], v['original_input_log_norm']) for v in (a, b)],
                                             'cosine_response_state_norm_rho': [rank_correlation(v['cosine'], v['original_state_log_norm']) for v in (a, b)]})
        print(json.dumps({'population': pop, 'response_rows': len(rows)}), flush=True)
    result = {'schema_version': 'signal-audit-response-refinement-v6',
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'decision': receipt('experiments/signal-audit-v6/iteration-4-decision.json'),
              'sources': sources, 'code': [receipt(__file__), receipt('src/sonnet_geometry/signal_audit.py')],
              'rows': rows, 'per_poem_measurements': details, 'new_model_calls': 0,
              'limits': 'Post-selection sensitivities with correlated nuisance controls and 24 poems; no p-values, '
                        'semantic labels or claim that all lexical/contextual explanations are controlled.',
              'elapsed_seconds': time.monotonic() - start}
    write_exclusive(out, canonical_json(result))
    print(json.dumps({'complete': True, 'output': receipt(out), 'seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
