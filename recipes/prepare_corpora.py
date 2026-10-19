"""Fetch credited sources and reconstruct exact model-input texts, failing on drift."""
import argparse,hashlib,json,sys,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sonnet_geometry.corpus import parse_gutenberg
from sonnet_geometry.confirmation_corpus import parse_sidney,select_population
from sonnet_geometry.spenser_corpus import parse_spenser

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('outputs/corpora'));parser.add_argument('--source-dir',type=Path,help='Use previously downloaded raw files instead of network');args=parser.parse_args()
 specs=json.loads((ROOT/'datasets/source-specs.json').read_text());args.output.mkdir(parents=True,exist_ok=True)
 for pop,spec in specs.items():
  filename=spec['url'].rsplit('/',1)[-1]
  if args.source_dir:raw=(args.source_dir/filename).read_bytes()
  else:
   request=urllib.request.Request(spec['url'],headers={'User-Agent':'TheShore-Sonnet-Research/1.0'})
   with urllib.request.urlopen(request,timeout=60) as response:raw=response.read()
  if hashlib.sha256(raw).hexdigest()!=spec['raw_sha256']:raise RuntimeError(f'{pop}: source bytes changed; do not substitute another edition')
  (args.output/filename).write_bytes(raw)
  if pop=='shakespeare':rows=parse_gutenberg(raw,retrieved_at='2026-10-18T04:15:58Z')
  elif pop=='sidney':rows=parse_sidney(raw,retrieved_at=None)[0];rows=select_population(rows)[0]
  else:rows=parse_spenser(raw,None)[0];rows=select_population(rows)[0]
  # parse_gutenberg returns the record list and its provenance receipt.
  if isinstance(rows,tuple):rows=rows[0]
  actual=[{k:r['sonnet_id'] if k=='poem_id' else r[k] for k in ('number','poem_id','text_sha256')} for r in rows]
  if actual!=spec['poems']:raise RuntimeError(f'{pop}: selected input texts differ')
  target=args.output/f'{pop}.jsonl';target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
  print(json.dumps({'population':pop,'poems':len(rows),'raw_sha256_verified':True,'all_input_text_hashes_verified':True,'output':str(target)}))
if __name__=='__main__':main()
