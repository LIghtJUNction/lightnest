from __future__ import annotations
import argparse,ctypes,hashlib,json,platform,random,statistics,subprocess,tempfile,time
from pathlib import Path
from template_io import chunk_packet,restore_chunks

def digest(b):return hashlib.sha256(b).hexdigest()
def run(command):
 t=time.perf_counter();p=subprocess.run(command,capture_output=True,check=True);return time.perf_counter()-t

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out;out.mkdir(exist_ok=True,parents=True)
 sources=[]
 for pth in ['/usr/lib/libz.so.1','/usr/lib/x86_64-linux-gnu/libz.so.1.3.1','/opt/pyvenv/lib/python3.13/site-packages/casadi/libz.so.1.3.1']:
  p=Path(pth);b=p.read_bytes();lib=ctypes.CDLL(str(p));lib.zlibVersion.restype=ctypes.c_char_p
  sources.append(dict(path=pth,sha256=digest(b),bytes=len(b),version=lib.zlibVersion().decode()))
 # This amendment is written BEFORE any patch/compression measurement.
 amendment=dict(reason='Official archive download failed: container DNS unavailable. Use installed known zlib libraries, without rebuilding or executing patches.',
  original_plan='same-toolchain zlib 1.3.1 to 1.3.2',replacement='installed 1.2.11 to 1.3.1 and two installed 1.3.1 builds',
  source_inputs=sources,limitations=['Not matched compiler options; build and library version effects are confounded.','No third-party full binary redistributed.','Local files read; only zlibVersion of already installed libraries was called.'])
 (out/'real_corpus_amendment.json').write_text(json.dumps(amendment,indent=2)+'\n')
 old,sysnew,alt=[Path(s['path']).read_bytes() for s in sources]
 rr=random.Random(314159)
 cases=[('installed_zlib_1.2.11_to_1.3.1',old,sysnew,'natural-version-build-difference'),
        ('installed_zlib_1.3.1_two_builds',sysnew,alt,'natural-build-difference'),
        ('one_byte_prefix_shift',sysnew,b'X'+sysnew,'synthetic byte-stream alignment stress; not a valid executable or a repair'),
        ('unrelated_random_128KiB',rr.randbytes(131072),rr.randbytes(131072),'negative compression control')]
 result=[]
 with tempfile.TemporaryDirectory() as tmp:
  d=Path(tmp);bp=d/'base';tp=d/'target';full=d/'full.zst';delta=d/'delta.zst';rest=d/'restored'
  for name,base,target,label in cases:
   bp.write_bytes(base);tp.write_bytes(target)
   times={'full_encode':[],'delta_encode':[],'full_decode':[],'delta_decode':[]}
   for _ in range(5):
    times['full_encode'].append(run(['zstd','-q','-f','-3','--single-thread',str(tp),'-o',str(full)]))
    times['delta_encode'].append(run(['zstd','-q','-f','-3','--single-thread','--mmap-dict','--patch-from='+str(bp),str(tp),'-o',str(delta)]))
    times['full_decode'].append(run(['zstd','-q','-f','-d',str(full),'-o',str(rest)]));assert rest.read_bytes()==target
    times['delta_decode'].append(run(['zstd','-q','-f','-d','--mmap-dict','--patch-from='+str(bp),str(delta),'-o',str(rest)]));assert rest.read_bytes()==target
   variants=[]
   for kind in ['fixed','gear']:
    t=time.perf_counter();packet,stats=chunk_packet(base,target,kind);stats['python_encode_seconds']=time.perf_counter()-t
    assert restore_chunks(base,packet)==target;stats['roundtrip']=True;variants.append(stats)
   record=dict(name=name,scope=label,base_bytes=len(base),target_bytes=len(target),base_sha256=digest(base),target_sha256=digest(target),
    zstd_full_bytes=full.stat().st_size,zstd_patch_bytes=delta.stat().st_size,
    patch_plus_base_target_hash_bytes=delta.stat().st_size+64,
    cold_full_base_then_patch_bytes=base.__len__()+delta.stat().st_size+64,
    timings_with_process_start_and_warm_os_cache=times,median_seconds={k:statistics.median(v) for k,v in times.items()},
    chunks=variants,all_roundtrips_match=True,patches_executed=False)
   result.append(record);print(json.dumps(record),flush=True)
 (out/'real_binary_results.json').write_text(json.dumps(dict(cases=result,sources=sources,zstd=subprocess.check_output(['zstd','--version'],text=True).strip()),indent=2)+'\n')
if __name__=='__main__':main()
