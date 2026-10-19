"""Verify public numeric artifacts and recreate selected paper results on CPU.

No model download, network access, cloud credential, or paid call is used.
The calculation path uses SciPy rather than the project's metric kernels.
"""
import argparse,hashlib,json
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1]
SEEDS=list(range(2026101801,2026101806));FAMILIES=('qwen','gemma','muse')

def rho(a,b):
 a,b=np.asarray(a),np.asarray(b)
 a,b=rankdata(np.round(a,12)),rankdata(np.round(b,12))
 if np.ptp(a)==0 or np.ptp(b)==0:return None
 return float(np.corrcoef(a,b)[0,1])
def unit(x):return x/np.linalg.norm(x,axis=1,keepdims=True)
def retrieval(x,y):
 scores=np.round(unit(x)@unit(y).T,12)
 return float(np.mean([np.mean([float(row[i]==np.max(row))/np.count_nonzero(row==np.max(row)) for i,row in enumerate(m)]) for m in (scores,scores.T)]))
def median(xs):return float(np.median(xs))
def distance(x):return pdist(x,metric='cosine')
def condition(c,s=None,p='fp32'):return f'{c}'+(f'_seed{s}' if s is not None else '')+f'_padded_{p}'

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('outputs/public-verification.json'));parser.add_argument('--figures',action='store_true');parser.add_argument('--data-dir',type=Path,default=ROOT/'data');args=parser.parse_args();data_dir=args.data_dir
 checks=[]
 def check(name,actual,expected,tol=1e-10):
  okay=actual is None and expected is None
  if actual is not None and expected is not None:okay=bool(np.allclose(actual,expected,atol=tol,rtol=0))
  checks.append({'check':name,'passes':okay,'actual':actual,'expected':expected})
  if not okay:raise AssertionError(checks[-1])
 manifest=json.loads((data_dir/'manifest.json').read_text())
 for row in manifest['files']:
  p=data_dir/Path(row['path']).name;assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],row['path']
 ref=json.loads((data_dir/'selected-reference-results.json').read_text())
 frozen=json.loads((data_dir/'frozen-claims.json').read_text())
 rdms=np.load(data_dir/'frozen-rdms.npz',allow_pickle=False)
 ix=json.loads((data_dir/'frozen-rdm-index.json').read_text())
 index={(r['population'],r['model'],r['condition'],r['state'],r['pooling']):rdms[r['array_key']] for r in ix}
 vectors=np.load(data_dir/'identity-final-means.npz',allow_pickle=False)
 def D(pop,f,c,state='final_raw',pool='mean'):return index[pop,f,c,state,pool]
 identity=[]
 for f in FAMILIES:
  for pop,train in [('sidney','spenser'),('spenser','sidney')]:
   x=vectors[f'{pop}.{f}.{condition("original")}'];tx=vectors[f'{train}.{f}.{condition("original")}']
   rows=[]
   for seed in SEEDS:
    y=vectors[f'{pop}.{f}.{condition("token_shuffle",seed)}'];ty=vectors[f'{train}.{f}.{condition("token_shuffle",seed)}']
    cx=unit(x)-unit(tx).mean(0);cy=unit(y)-unit(ty).mean(0)
    rows.append({'seed':seed,'raw_identity':retrieval(x,y),'calibrated_identity':retrieval(cx,cy),'raw_rdm_rho':rho(distance(x),distance(y)),'centered_rdm_rho':rho(distance(cx),distance(cy))})
   selected=next(r for r in ref['selected_cross_author_identity'] if r['model']==f and r['test_population']==pop)
   geom=next(r for r in ref['selected_calibrated_relational_geometry'] if r['model']==f and r['test_population']==pop)
   for i,row in enumerate(rows):
    check(f'identity/{pop}/{f}/{row["seed"]}/raw',row['raw_identity'],selected['scores']['raw']['top1']['values'][i])
    check(f'identity/{pop}/{f}/{row["seed"]}/calibrated',row['calibrated_identity'],selected['scores']['calibration_centered']['top1']['values'][i])
    check(f'geometry/{pop}/{f}/{row["seed"]}/raw',row['raw_rdm_rho'],geom['methods']['raw_cosine']['rdm_rho']['values'][i])
    check(f'geometry/{pop}/{f}/{row["seed"]}/centered',row['centered_rdm_rho'],geom['methods']['separate_unit_centers']['rdm_rho']['values'][i])
   identity.append({'population':pop,'model':f,'seeds':rows,'medians':{k:median([r[k] for r in rows]) for k in rows[0] if k!='seed'}})
 frozen_results=[]
 for pop in ('sidney','spenser'):
  for f in FAMILIES:
   early=[rho(D(pop,f,condition('original'),'early'),D(pop,f,condition('token_shuffle',s),'early')) for s in SEEDS]
   final=[rho(D(pop,f,condition('original')),D(pop,f,condition('token_shuffle',s))) for s in SEEDS]
   changes=[a-b for a,b in zip(early,final)]
   passed=median(early)>=.8 and median(changes)>=.3 and min(changes)>0
   claim=next(r for r in frozen[pop]['claims'] if r.get('model')==f)
   for name,actual in [('early_stability',early),('final_raw_stability',final),('early_minus_final_raw',changes)]:
    for metric,fn in [('median',median),('minimum',min),('maximum',max)]:check(f'frozen/{pop}/{f}/{name}/{metric}',fn(actual),claim[name][metric])
   check(f'frozen/{pop}/{f}/decision',passed,claim['passes'])
   frozen_results.append({'population':pop,'model':f,'early':early,'final':final,'paired_changes':changes,'passes':passed})
  for left,right in combinations(FAMILIES,2):
   original=rho(D(pop,left,condition('original')),D(pop,right,condition('original')))
   shuffled=[rho(D(pop,left,condition('token_shuffle',s)),D(pop,right,condition('token_shuffle',s))) for s in SEEDS]
   changes=[v-original for v in shuffled]
   claim=next(r for r in frozen[pop]['claims'] if r.get('models')==[left,right])
   # Historical key names are retained in the public reference.
   field='shuffled_minus_original'
   check(f'agreement/{pop}/{left}/{right}/median',median(changes),claim[field]['median'])
   check(f'agreement/{pop}/{left}/{right}/decision',median(changes)>=.2 and min(changes)>0,claim['passes'])
   frozen_results.append({'population':pop,'models':[left,right],'original':original,'shuffled':shuffled,'changes':changes,'passes':claim['passes']})
 structural=[]
 for selected in ref['selected_structural_contrasts']:
  pop,f,state=selected['population'],selected['model'],selected['state'];pool=selected['pooling']
  values=[];omits=[]
  n=24;tri=np.triu_indices(n,1)
  for seed in SEEDS:
   a,b,c=[D(pop,f,condition(control,None if control=='original' else seed),state,pool) for control in ('original','line_shuffle','token_shuffle')]
   values.append(rho(a,b)-rho(a,c))
   for omit in range(24):
    mask=(tri[0]!=omit)&(tri[1]!=omit);omits.append(rho(a[mask],b[mask])-rho(a[mask],c[mask]))
  for scope,values_ in [('full',values),('omissions',omits)]:
   for metric,fn in [('minimum',min),('maximum',max),('median',median)]:check(f'structure/{pop}/{f}/{state}/{scope}/{metric}',fn(values_),selected['line_minus_token'][scope][metric])
  structural.append({'population':pop,'model':f,'state':state,'seed_differences':values,'omission_range':[min(omits),max(omits)]})
 responses=json.loads((data_dir/'response-displacements.json').read_text())
 ri={(r['population'],r['model'],r['precision'],r['intervention'],r['seed'],r['state'],r['pooling']):r for r in responses}
 components=[]
 for selected in ref['selected_response_components']:
  pop=selected['population'];left,right=selected['models'];state=selected['state'];pool=selected['pooling'];control=selected['intervention']
  a,b=[np.stack([ri[pop,f,'fp32',control,s,state,pool]['per_poem_cosine_displacement'] for s in SEEDS],axis=1) for f in (left,right)]
  same=[rho(a[:,i],b[:,i]) for i in range(5)]
  mean=rho(a.mean(1),b.mean(1));ar,br=[rankdata(np.round(x,12),axis=1) for x in (a,b)]
  within=rho(ar.ravel(),br.ravel())
  check(f'response/{pop}/{left}/{right}/same',median(same),selected['same_seed']['median'])
  check(f'response/{pop}/{left}/{right}/poem',mean,selected['mean_across_seeds_poem_response_rho'])
  check(f'response/{pop}/{left}/{right}/within',within,selected['within_poem_seed_rank_correlation'])
  components.append({'population':pop,'models':[left,right],'same_seed':median(same),'poem_average':mean,'within_poem':within})
 result={'schema_version':'public-verification-v1','all_pass':all(r['passes'] for r in checks),'numeric_checks':len(checks),'file_checks':len(manifest['files']),'identity':identity,'frozen':frozen_results,'structural':structural,'response':components,'checks':checks,'scope':'Recalculates selected quantitative results from compact derived data; does not repeat inference or all historical analyses.','new_model_calls':0}
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 if args.figures:plot(result,args.output.parent/'figures')
 print(json.dumps({k:result[k] for k in ('all_pass','numeric_checks','file_checks','new_model_calls')}))

def plot(result,out):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out.mkdir(parents=True,exist_ok=True)
 fig,ax=plt.subplots(figsize=(9,4.5));labels=[r['model']+' / '+r['population'] for r in result['identity']];x=np.arange(6)
 ax.bar(x-.18,[r['medians']['raw_identity'] for r in result['identity']],.36,label='Raw identity')
 ax.bar(x+.18,[r['medians']['calibrated_identity'] for r in result['identity']],.36,label='Other-author centered')
 ax.set_xticks(x,labels,rotation=25,ha='right');ax.set_ylim(0,1.2);ax.set_ylabel('Median bidirectional matching accuracy');ax.legend(loc='upper center',ncol=2);fig.tight_layout();fig.savefig(out/'identity-reproduced.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,4.5));labels=[r['population']+' / '+'-'.join(r['models']) for r in result['response']];x=np.arange(6)
 for i,key in enumerate(('same_seed','poem_average','within_poem')):ax.bar(x+(i-1)*.25,[r[key] for r in result['response']],.25,label=key.replace('_',' '))
 ax.set_xticks(x,labels,rotation=25,ha='right');ax.set_ylim(0,1.15);ax.set_ylabel('Descriptive rank correlation');ax.legend(loc='upper center',ncol=3);fig.tight_layout();fig.savefig(out/'response-reproduced.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
