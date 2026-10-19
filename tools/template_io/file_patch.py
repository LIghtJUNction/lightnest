"""Hash-bound Zstandard file deltas. Never executes generated files.

Commands write to fresh output paths. Decoding is bounded by declared output
size, a CLI size ceiling, zstd memory limit, CPU limit and wall timeout (POSIX).
This is a local research utility, not a reviewed multi-user service.
"""
from __future__ import annotations
import argparse,hashlib,json,os,resource,shutil,stat,subprocess
from pathlib import Path

def regular(path: Path) -> Path:
    path=path.absolute()
    if not stat.S_ISREG(path.lstat().st_mode):raise ValueError('regular non-symlink file required')
    return path

def file_sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def run(args,stdout=None,limit=None):
    def limits():
        os.umask(0o077)
        resource.setrlimit(resource.RLIMIT_CPU,(60,60))
        if limit is not None:resource.setrlimit(resource.RLIMIT_FSIZE,(limit,limit))
    subprocess.run(args,check=True,stdout=stdout,stderr=subprocess.PIPE,timeout=120,preexec_fn=limits)

def create(target:Path,out:Path,base:Path|None=None,min_saving:float=.10):
    target=regular(target);base=regular(base) if base is not None else None
    if not 0<=min_saving<1:raise ValueError('min_saving must be in [0,1)')
    out.mkdir(mode=0o700,parents=True,exist_ok=False)
    before_target=file_sha(target);before_base=file_sha(base) if base else None
    full=out/'full.tmp.zst';patch=out/'delta.tmp.zst';payload=out/'payload.zst'
    cmd=['zstd','-q','-3','--single-thread']
    run(cmd+[str(target),'-o',str(full)])
    chosen=full;kind='full'
    if base is not None:
        run(cmd+['--mmap-dict','-M128MB','--patch-from='+str(base),str(target),'-o',str(patch)])
        if patch.stat().st_size+32 < (1-min_saving)*full.stat().st_size:chosen=patch;kind='delta'
    if file_sha(target)!=before_target or (base and file_sha(base)!=before_base):raise ValueError('inputs changed during encoding')
    sizes={'full_zstd_bytes':full.stat().st_size,'delta_zstd_bytes':patch.stat().st_size if patch.exists() else None}
    chosen.rename(payload)
    for unused in [full,patch]:
        if unused.exists():unused.unlink()
    manifest={'version':1,'kind':kind,'payload':'payload.zst','payload_sha256':file_sha(payload),
      'target_sha256':before_target,'target_bytes':target.stat().st_size,'base_sha256':before_base if kind=='delta' else None,
      'sizes':sizes,'base_must_already_be_available':kind=='delta','executed':False}
    with (out/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    os.chmod(payload,0o600);os.chmod(out/'manifest.json',0o600)
    return manifest

def restore(package:Path,output:Path,base:Path|None=None,max_bytes:int=512<<20):
    manifest_path=regular(package/'manifest.json')
    if manifest_path.stat().st_size>65536:raise ValueError('manifest size limit')
    m=json.loads(manifest_path.read_text())
    if m.get('version')!=1 or m.get('payload')!='payload.zst' or m.get('kind') not in ('full','delta'):raise ValueError('manifest format')
    n=m.get('target_bytes')
    if type(n) is not int or not 0<=n<=max_bytes:raise ValueError('output limit')
    payload=regular(package/'payload.zst')
    if file_sha(payload)!=m['payload_sha256']:raise ValueError('payload digest mismatch')
    args=['zstd','-q','-d','-c','-M128MB']
    if m['kind']=='delta':
        if base is None:raise ValueError('base file required')
        base=regular(base)
        if file_sha(base)!=m['base_sha256']:raise ValueError('wrong base file')
        args+=['--mmap-dict','--patch-from='+str(base)]
    args+=[str(payload)]
    # Exclusive creation protects existing user files. A failed decode removes
    # only the newly created candidate, never the base or existing destination.
    fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(fd,'wb') as f:run(args,stdout=f,limit=max(1,n+1))
        if output.stat().st_size!=n or file_sha(output)!=m['target_sha256']:raise ValueError('target digest mismatch')
    except BaseException:
        output.unlink(missing_ok=True);raise
    return {'output_sha256':file_sha(output),'bytes':n,'executed':False}

def main():
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest='action',required=True)
    a=sp.add_parser('pack');a.add_argument('--target',type=Path,required=True);a.add_argument('--base',type=Path);a.add_argument('--out',type=Path,required=True)
    a.add_argument('--min-saving',type=float,default=.10)
    a=sp.add_parser('unpack');a.add_argument('--package',type=Path,required=True);a.add_argument('--base',type=Path);a.add_argument('--out',type=Path,required=True);a.add_argument('--max-bytes',type=int,default=512<<20)
    a=p.parse_args()
    if shutil.which('zstd') is None:raise RuntimeError('zstd CLI is required')
    result=create(a.target,a.out,a.base,a.min_saving) if a.action=='pack' else restore(a.package,a.out,a.base,a.max_bytes)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
