from __future__ import annotations
import argparse,gzip,hashlib,io,json,os,platform,subprocess,time,zipfile
from pathlib import Path
from itertools import zip_longest
from collections import Counter
from template_io import CoreRow,pack_core,unpack_core,make_delta,transport_packet,compression_template_key

def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=True)
 raw_archive=a.dataset.read_bytes();archive_hash=sha(raw_archive)
 if archive_hash!='787fddaaff143052deb63b4c634c5d23550e9060a0fd8fbe3304b5e5b1857f4c':raise ValueError('unexpected dataset version')
 z=zipfile.ZipFile(io.BytesIO(raw_archive));prefix='LightNest_scale_v2/LightNest_scale_data_v2/data/'
 manifest=json.loads(z.read(prefix+'manifest.json'));rows=[];shards_checked=0
 original_gzip_bytes=0;programs=Counter();full=Counter();shapes=Counter();inputsizes=[];outd=[];packetbytes=0;diffcount=Counter()
 t=time.perf_counter()
 for sh in manifest['shards']:
  if sh['split']!='train':continue
  streams=[]
  for name in ['inputs','labels']:
   raw=z.read(prefix+sh[name]['path'])
   if sha(raw)!=sh[name]['sha256']:raise ValueError('source shard changed')
   original_gzip_bytes+=len(raw);streams.append(io.StringIO(gzip.decompress(raw).decode()))
  shards_checked+=1
  for left,right in zip_longest(*streams):
   if left is None or right is None:raise ValueError('unaligned tables')
   inp,lab=json.loads(left),json.loads(right)
   if inp['record_id']!=lab['record_id']:raise ValueError('id mismatch')
   b=bytes.fromhex(inp['program_hex']);target=bytes.fromhex(lab['target_hex'])
   row=CoreRow(bytes.fromhex(inp['record_id']),inp['request'],b,target);rows.append(row)
   programs[b]+=1;full[(row.command,b)]+=1;shapes[compression_template_key(b)]+=1
   inputsizes.append(len(b));d=make_delta(b,target);outd.append(len(d));packetbytes+=len(transport_packet(b,target))
   diffcount[sum(x!=y for x,y in zip(b,target))]+=1
 readseconds=time.perf_counter()-t
 if len(rows)!=109272:raise ValueError('unexpected record count')
 stat={};packed={}
 for name,templates in [('raw',False),('template_delta',True)]:
  t=time.perf_counter();payload,stat[name]=pack_core(rows,templates);stat[name]['encode_seconds']=time.perf_counter()-t
  stat[name]['sha256']=sha(payload);path=a.out/(name+'.core');path.write_bytes(payload);packed[name]=payload
  t=time.perf_counter();n=0
  for old,new in zip_longest(rows,unpack_core(payload)):
   if old!=new:raise AssertionError('roundtrip mismatch')
   n+=1
  stat[name]['decode_and_compare_seconds']=time.perf_counter()-t;stat[name]['verified_records']=n
  times=[]
  for _ in range(3):
   t=time.perf_counter();subprocess.run(['zstd','-q','-f','-3','--single-thread',str(path),'-o',str(path)+'.zst'],check=True);times.append(time.perf_counter()-t)
  comp=Path(str(path)+'.zst').read_bytes();stat[name]['zstd3_bytes']=len(comp);stat[name]['zstd3_seconds']=times
  restored=subprocess.check_output(['zstd','-q','-d','-c',str(path)+'.zst'])
  assert restored==payload
  stat[name]['compressed_roundtrip']=True
 sizes=sorted(inputsizes)
 result=dict(dataset_sha256=archive_hash,train_only=True,heldout_records_read=0,records=len(rows),source_shards_hash_checked=shards_checked,
  unique_binary_inputs=len(programs),unique_exact_requests=len(full),unique_operand_normalized_compression_templates=len(shapes),
  input_bytes=sum(inputsizes),target_bytes=sum(len(r.target) for r in rows),changed_byte_histogram=dict(sorted(diffcount.items())),
  compact_output_delta_bytes=sum(outd),standalone_hash_bound_output_packet_bytes=packetbytes,
  input_size_min=sizes[0],input_size_median=sizes[len(sizes)//2],input_size_max=sizes[-1],
  original_inputs_and_labels_gzip_bytes=original_gzip_bytes,source_read_seconds=readseconds,core=stat,
  zstd_version=subprocess.check_output(['zstd','--version'],text=True).strip(),python=platform.python_version(),
  limitations=['Template encoding removes no information; every original request,input,target,record ID restored.',
   'Core sizes exclude execution/provenance metadata on both sides and cannot be compared to whole dataset ZIP size.',
   'Output deltas are computed from supplied dataset labels, not new model predictions; no generation accuracy claimed.',
   'Input compression template selection reads only input bytes. Templates are not semantic or neural cache keys.',
   'Dataset records include correlated partial repair states, not independent programs.'])
 (a.out/'dataset_results.json').write_text(json.dumps(result,indent=2)+'\n')
 # Deterministic input-only selection. Do not inspect correctness when choosing.
 selected=sorted(full,key=lambda x:sha(x[0].encode()+x[1]))[:16]
 (a.out/'inference_inputs.json').write_text(json.dumps([{'command':c,'binary':b.hex()} for c,b in selected],indent=2)+'\n')
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
