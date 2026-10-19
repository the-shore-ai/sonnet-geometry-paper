"""Descriptive comparison of two complete, identically specified poem populations.

No statistical independence, author-causal effect, or literary usefulness is inferred.
Run after strict per-population analysis; never selects seeds or changes criteria.
"""
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from sonnet_geometry.storage import canonical_json,sha256_file,write_exclusive

OLD=Path('outputs/confirmation-v4/muse-finalization-repair/strict-analysis.json')
NEW=Path('outputs/confirmation-v5-spenser/strict-analysis.json')

def receipt(path):return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha256_file(path)}

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--sidney-analysis',type=Path,default=OLD);parser.add_argument('--spenser-analysis',type=Path,default=NEW);parser.add_argument('--output',type=Path,default=Path('experiments/confirmation-v5-spenser/population-comparison.json'));args=parser.parse_args()
 old_path,new_path=args.sidney_analysis,args.spenser_analysis
 old,new=[json.loads(p.read_text()) for p in [old_path,new_path]]
 assert old['schema_version']==new['schema_version']=='depth-confirmation-analysis-v4'
 assert old['criteria']==new['criteria'] and old['poem_count']==new['poem_count']==24
 assert all(m['execution_complete'] is True for a in [old,new] for m in a['models'].values())
 protocols=[json.loads(Path(a['protocol']['path']).read_text()) for a in [old,new]]
 assert protocols[0]['population']['author']=='Philip Sidney' and protocols[1]['population']['author']=='Edmund Spenser'
 assert protocols[1]['study_namespace']=='confirmation-v5-spenser'
 assert protocols[1]['extension']['parent_protocol_sha256']==old['protocol']['sha256']
 for a,p in zip([old,new],protocols,strict=True):assert a['protocol']['sha256']==sha256_file(Path(a['protocol']['path']))
 for field in ['analysis','conditions','model_locks','states','poolings','pad_length','randomization']:assert protocols[0][field]==protocols[1][field]
 populations=[]
 for a,p in zip([old,new],protocols,strict=True):
  rows=[]
  for c in a['claims']:
   identity=c.get('model','+'.join(c.get('models',[])))
   if 'model' in c:
    row={'subject':identity,'role':c['role'],'early_median':c['early_stability']['median'],
         'final_raw_median':c['final_raw_stability']['median'],'paired_delta':c['early_minus_final_raw']}
   else:
    candidates=[r for r in a['cross_model'] if r['models']==c['models'] and r['name']=='original_padded_fp32' and r['state']=='final_raw' and r['pooling']=='mean'];assert len(candidates)==1
    row={'subject':identity,'role':c['role'],'original_final_raw_agreement':candidates[0]['rho'],'paired_delta':c['shuffled_minus_original']}
   row.update(passes=c['passes'],omission_directions=c['omission_direction_sensitivity'],sensitivity_qualified_status=c['sensitivity_qualified_status'])
   rows.append(row)
  populations.append({'author':p['population']['author'],'edition':p['population']['edition'],'poems':a['poems'],
                      'status':a['status'],'claims':rows})
 result={'schema_version':'sidney-spenser-descriptive-comparison-v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
 'sources':[receipt(old_path),receipt(new_path),receipt(Path(__file__))],
 'scientific_fields_identical':['analysis','conditions','model_locks','states','poolings','pad_length','randomization'],
 'populations':populations,'comparative_significance_tests':0,
 'limits':['Author, edition, spelling, length distributions and sample differ together; no author-causal attribution.',
           'Five seeds and 120 omission contrasts per criterion are dependent sensitivities, not independent population samples or confidence intervals.',
           'Third poet chosen after Sidney outcomes were known; thresholds and selected inputs frozen before new inference.',
           'The earlier12-poem Shakespeare control uses a different selected population and precision design; its figures are not merged into this matched-FP32 comparison.',
           'No human usefulness evidence has been collected.']}
 write_exclusive(args.output,canonical_json(result))
 for pop in populations:
  print(pop['author'],[(r['subject'],None if r['paired_delta']['median'] is None else round(r['paired_delta']['median'],4),r['passes']) for r in pop['claims']])
if __name__=='__main__':main()
