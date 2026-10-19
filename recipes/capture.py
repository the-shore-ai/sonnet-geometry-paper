"""Local extraction wrapper around the exact published historical capture kernels.

Planning is the default. --execute uses the caller's existing hardware; no
Modal/GCP account or cloud submission is involved. This new wrapper was not
the historical scheduler; its status is explicit in every new run report.
"""
import argparse,hashlib,importlib.util,json,platform,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sonnet_geometry.extraction import load_model,extract_sonnet
from sonnet_geometry.confirmation_inputs import build_cases,audit_cases
from sonnet_geometry.text_controls import capture_selected,pool_selected,EARLY_BLOCK,DEPTH75_BLOCK,LAST_BLOCK

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--population',choices=['shakespeare','sidney','spenser'],required=True);p.add_argument('--model',choices=['qwen','gemma','muse'],required=True);p.add_argument('--corpus-root',type=Path,default=Path('outputs/corpora'));p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--all-cases',action='store_true');p.add_argument('--max-cases',type=int,default=1);p.add_argument('--retain-token-states',action='store_true');a=p.parse_args()
 if a.max_cases<1:raise ValueError('Positive max-cases required')
 records=[json.loads(s) for s in (a.corpus_root/f'{a.population}.jsonl').read_text().splitlines()]
 expected=json.loads((ROOT/'datasets/source-specs.json').read_text())[a.population]['poems']
 actual=[{'number':r['number'],'poem_id':r['sonnet_id'],'text_sha256':hashlib.sha256(r['text'].encode()).hexdigest()} for r in records]
 if actual!=expected:raise ValueError('Corpus texts/order differ from the registered inputs')
 locks=json.loads((ROOT/'historical/protocols/sidney.json').read_text())['model_locks'];lock=locks[a.model]
 print(json.dumps({'execute':a.execute,'model':a.model,'checkpoint':lock,'poems':len(records),'planned_limit':'all' if a.all_cases else a.max_cases,'capture':'all native residual depths' if a.population=='shakespeare' else 'five states; three poolings; paired precision','requires_own_hardware':True,'new_cloud_submissions':0}))
 if not a.execute:return
 import torch
 from safetensors.torch import save_file as save_torch
 from safetensors.numpy import save_file as save_numpy
 if not torch.cuda.is_available():raise RuntimeError('Existing CUDA device required for these checkpoint recipes')
 if a.output.exists():raise FileExistsError('Choose a new run directory')
 a.output.mkdir(parents=True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 model,tokenizer=load_model(lock['model_id'],lock['revision'],dtype='bfloat16',device='cuda',attention_implementation='eager')
 start=time.monotonic();reports=[];pooled={}
 if a.population=='shakespeare':
  layers={'qwen':[0,18,35],'gemma':[0,30,58],'muse':[0,26,51]}[a.model]
  for i,r in enumerate(records):
   if not a.all_cases and i>=a.max_cases:break
   run=extract_sonnet(model,tokenizer,r,a.output,model_id=lock['model_id'],revision=lock['revision'],capture=['q_proj','k_proj','v_proj','attention_output','mlp_intermediate','mlp_output'],capture_layers=layers,run_id=f'public-{a.model}-{r["number"]:03d}')
   reports.append({'poem':r['number'],'run_id':run['run_id']})
 else:
  spec=importlib.util.spec_from_file_location('historical_confirmation',ROOT/'historical/infra/modal_confirmation.py');historical=importlib.util.module_from_spec(spec);sys.modules[spec.name]=historical;spec.loader.exec_module(historical)
  protocol=json.loads((ROOT/f'historical/protocols/{a.population}.json').read_text());cases=[c for r in records for c in build_cases(r,tokenizer,pad_length=256)];audit_cases(cases)
  remaining=a.max_cases;blocks={'early':EARLY_BLOCK[a.model],'depth75':DEPTH75_BLOCK[a.model],'final_raw':LAST_BLOCK[a.model]}
  original=None
  for phase,precision,phase_cases in historical.execution_phases(protocol,{'cases':cases}):
   if precision=='fp32':original=historical.promote_precision(model)
   for c in phase_cases:
    if not a.all_cases and remaining<=0:break
    states=capture_selected(model,c['tokenization'],blocks=blocks,pad_to=256,pad_id=c['pad_token_id']);values=pool_selected(states,c['tokenization'])
    for k,v in values.items():pooled[f'{c["sonnet"]:03d}.{c["condition"]}.{k}']=v
    if a.retain_token_states:save_torch(states,str(a.output/f'{c["sonnet"]:03d}.{c["condition"]}.safetensors'))
    reports.append({'poem':c['sonnet'],'condition':c['condition'],'forward_sha256':c['forward_sha256'],'token_count':len(c['tokenization']['input_ids']),'parameter_dtypes':sorted({str(x.dtype) for x in model.parameters()})});remaining-=1
   if original is not None:historical.restore_precision(model,original);original=None
   if not a.all_cases and remaining<=0:break
  save_numpy(pooled,str(a.output/'pooled.safetensors'))
 report={'schema_version':'public-recipe-capture-v1','complete':a.all_cases,'historical_scheduler':False,'kernel_manifest':'provenance/original-source-manifest.json','model':lock,'population':a.population,'cases':reports,'runtime':{'torch':torch.__version__,'python':platform.python_version(),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()},'seconds':time.monotonic()-start,'new_cloud_submissions':0}
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'captured_cases':len(reports),'output':str(a.output)}))
if __name__=='__main__':main()
