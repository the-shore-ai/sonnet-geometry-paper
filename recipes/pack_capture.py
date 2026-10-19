"""Rebuild compact numerical inputs from six complete local pooled captures.

Reference-result files remain the published expectations, not new outcomes.
No network, model inference, or private credentials are used by this step.
"""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sonnet_geometry.signal_audit import unit_rows
from sonnet_geometry.text_control_analysis import key,cosine_rdm
from sonnet_geometry.confirmation_inputs import conditions

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Choose an empty compact output path')
 a.output.mkdir(parents=True);reference=json.loads((ROOT/'data/frozen-claims.json').read_text());specs=conditions()
 # Carry only unchanged input-design metadata. All displacement/norm arrays are recomputed.
 response_reference=json.loads((ROOT/'data/response-displacements.json').read_text())
 input_metadata={(r['population'],r['model'],r['precision'],r['intervention'],r['seed'],r['state'],r['pooling']):r['per_poem_changed_token_fraction'] for r in response_reference}
 rdms={};indices=[];identity={};responses=[];sources=[]
 for pop in ('sidney','spenser'):
  ids=reference[pop]['poems'];rdm_number=0
  for f in ('qwen','gemma','muse'):
   source=a.runs_root/pop/f/'pooled.safetensors'
   sources.append({'population':pop,'model':f,'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'bytes':source.stat().st_size})
   with safe_open(source,framework='numpy') as h:
    matrices={}
    for condition in specs:
     name=condition['name']
     for state in ('input','early','depth75','final_raw','final_normalized'):
      for pool in ('mean','last','last_lexical'):
       x=np.stack([h.get_tensor(key(n,name,state,pool)) for n in ids]);assert x.dtype==np.float64 and np.isfinite(x).all()
       matrices[name,state,pool]=x;array_key=f'{pop}.{rdm_number:04d}';rdm_number+=1
       rdms[array_key]=cosine_rdm(x)[np.triu_indices(24,1)]
       indices.append({'population':pop,'model':f,'condition':name,'state':state,'pooling':pool,'array_key':array_key})
       if state=='final_raw' and pool=='mean' and condition['precision']=='fp32' and condition['intervention'] in ('original','token_shuffle'):identity[f'{pop}.{f}.{name}']=x
    for precision in ('bf16','fp32'):
     for control in ('word_shuffle','line_shuffle','token_shuffle'):
      for state in ('input','early','depth75','final_raw','final_normalized'):
       for pool in ('mean','last','last_lexical'):
        x=matrices[f'original_padded_{precision}',state,pool];ux=unit_rows(x)
        for seed in range(2026101801,2026101806):
         y=matrices[f'{control}_seed{seed}_padded_{precision}',state,pool];uy=unit_rows(y)
         responses.append({'population':pop,'model':f,'precision':precision,'intervention':control,'seed':seed,'state':state,'pooling':pool,'poems':ids,'per_poem_cosine_displacement':np.clip(1-np.sum(ux*uy,axis=1),0,2).tolist(),'per_poem_log_norm_ratio':np.log(np.linalg.norm(y,axis=1)/np.linalg.norm(x,axis=1)).tolist(),'per_poem_changed_token_fraction':input_metadata[pop,f,precision,control,seed,state,pool]})
 def save(name,obj):(a.output/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
 np.savez_compressed(a.output/'identity-final-means.npz',**identity);np.savez_compressed(a.output/'frozen-rdms.npz',**rdms);save('frozen-rdm-index.json',indices);save('response-displacements.json',responses)
 for name in ('frozen-claims.json','selected-reference-results.json','complete-exploratory-summary.json'):shutil.copyfile(ROOT/'data'/name,a.output/name)
 files=[{'path':'data/'+f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for f in sorted(a.output.iterdir())]
 save('manifest.json',{'schema_version':'public-regenerated-compact-v1','files':files,'numeric_sources':sources,'scope':'Vectors/RDMs/response magnitudes recomputed from provided pooled captures. Expected-result files and registered token-change fractions remain original published reference material. Capture provenance must be checked separately.','new_model_calls':0})
 print(json.dumps({'arrays':len(rdms),'responses':len(responses),'output':str(a.output),'new_model_calls':0}))
if __name__=='__main__':main()
