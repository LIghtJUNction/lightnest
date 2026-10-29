"""Validate generated scan functions in a bounded native child process.

Run after behavior_space_study.py. This is a post-study backend diagnostic,
not another training run. It tests mask 0 / 32 initial + 32 active labels.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lightnest_rgb.loop_lift import emit_scan, emit_demo_elf


def pack_cases(cases: list[bytes]) -> bytes:
    return struct.pack("<I",len(cases))+b"".join(struct.pack("<I",len(x))+x for x in cases)


def expected(table: np.ndarray, data: bytes, mode: str) -> int:
    bits=table[np.frombuffer(data,np.uint8)]
    if mode=="count":
        return int(bits.sum())
    stop=bits!=0 if mode=="first" else bits==0
    ids=np.flatnonzero(stop)
    return int(ids[0]) if len(ids) else len(data)


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().with_name("behavior_native_runner.c")
    runner=args.out/"runner"
    subprocess.run(["gcc","-O2","-std=c11","-Wall","-Wextra","-Werror",
                    str(source),"-o",str(runner)],check=True,capture_output=True)
    truths=np.load(args.study/"data.npz",allow_pickle=False)["test"]
    rows=[json.loads(line) for line in (args.study/"predictions.jsonl").read_text().splitlines()]
    selected=[r for r in rows if (r["policy"],r["initial"],r["extra"],r["mask"])==
              ("entropy_learned",32,32,0)]
    if len(selected)!=len(truths):
        raise ValueError("expected exactly one fixed-mask candidate per target")
    libc=ctypes.CDLL(None)
    for name in ("strspn","strcspn"):
        fn=getattr(libc,name);fn.argtypes=[ctypes.c_char_p,ctypes.c_char_p];fn.restype=ctypes.c_size_t
    results=[]
    demo_saved=False
    for row in selected:
        i=row["target"];truth=truths[i]
        pred=np.unpackbits(np.frombuffer(bytes.fromhex(row["prediction_hex"]),np.uint8))
        pos=np.flatnonzero(truth);neg=np.flatnonzero(truth==0)
        a=int(pos[0]);b=int(neg[0])
        rng=np.random.default_rng(720000+i)
        cases=[b""]+[bytes([x]) for x in range(256)]
        for n in (2,3,7,15,16,31,32,255,256,257,4096,8192,131072):
            cases.append(rng.integers(0,256,n,dtype=np.uint8).tobytes())
        cases += [bytes([a])*131072, bytes([b])*131071+bytes([a]),
                  bytes([a])*131071+bytes([b]), bytes([b])*131072]
        # Separate non-NUL strings allow comparison with actual libc functions.
        string_cases=[rng.integers(1,256,n,dtype=np.uint8).tobytes() for n in (0,1,16,256,4096)]
        cases += string_cases
        folder=args.out/f"target-{i:03d}";folder.mkdir()
        data=pack_cases(cases);(folder/"cases.bin").write_bytes(data)
        accept=bytes(int(x) for x in pos if x!=0)
        for mode in ("count","first","prefix"):
            code=emit_scan(pred.tobytes(),mode)
            binary=folder/f"{mode}.bin";binary.write_bytes(code)
            run=subprocess.run([str(runner),str(binary),str(folder/"cases.bin")],
                               capture_output=True,timeout=3)
            (folder/f"{mode}.stdout").write_bytes(run.stdout)
            if run.returncode!=0 or len(run.stdout)!=8*len(cases):
                raise RuntimeError(f"native runner failure {i}/{mode}: {run.returncode}")
            got=np.frombuffer(run.stdout,dtype="<u8")
            want=np.array([expected(truth,c,mode) for c in cases],dtype=np.uint64)
            encoded_want=np.array([expected(pred,c,mode) for c in cases],dtype=np.uint64)
            backend_ok=bool(np.array_equal(got,encoded_want))
            if not backend_ok:
                raise AssertionError("machine code disagrees with encoded predicted predicate")
            libc_ok=None
            if mode!="count":
                fn=libc.strcspn if mode=="first" else libc.strspn
                libc_values=np.array([fn(c,accept) for c in string_cases],np.uint64)
                if not np.array_equal(libc_values,want[-len(string_cases):]):
                    raise AssertionError("independent libc reference mismatch")
                libc_ok=bool(np.array_equal(got[-len(string_cases):],libc_values))
            record=dict(target=i,mode=mode,cases=len(cases),body_bytes=len(code),
                        max_input_length=max(map(len,cases)),returncode=run.returncode,
                        exact_behavior=bool(np.array_equal(pred,truth)),
                        all_reference_outputs_correct=bool(np.array_equal(got,want)),
                        backend_matches_encoded_predicate=backend_ok,
                        libc_string_tests_correct=libc_ok,
                        sha256=hashlib.sha256(code).hexdigest(),
                        output_sha256=hashlib.sha256(run.stdout).hexdigest(),
                        failed_case_indices=np.flatnonzero(got!=want).tolist())
            results.append(record)
            if mode=="count" and record["exact_behavior"] and not demo_saved:
                demo=args.out/"demo";demo.mkdir()
                input_bytes=bytes(range(256))*16
                elf=emit_demo_elf(pred.tobytes(),mode,input_bytes)
                path=demo/"program.elf";path.write_bytes(elf);path.chmod(0o700)
                # Only trusted encoder output is run directly; arbitrary code uses the runner.
                e=subprocess.run([str(path)],capture_output=True,timeout=3)
                path.chmod(0o600)
                value=expected(truth,input_bytes,mode)
                if e.returncode!=0 or e.stdout!=struct.pack("<Q",value):
                    raise AssertionError("standalone ELF demo mismatch")
                meta=dict(target=i,mode=mode,initial_examples=32,external_queries=32,
                          embedded_input_bytes=len(input_bytes),expected=value,
                          elf_bytes=len(elf),exit_code=e.returncode,
                          verified_for_demo_input=True,arbitrary_input_proof=False,
                          generated_by="inferred table plus trusted hand-written loop and ELF backend")
                (demo/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
                demo_saved=True
    summary=dict(functions=len(results),cases=sum(r["cases"] for r in results),
                 complete_correct_functions=sum(r["all_reference_outputs_correct"] for r in results),
                 backend_mismatches=sum(not r["backend_matches_encoded_predicate"] for r in results),
                 max_input_length=131072,unique_targets=len(truths),
                 source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 selection="fixed mask0, 32 initial +32 active labels; post-study backend diagnostic",
                 note="singleton tests are exhaustive over bytes; arrays are finite stress tests, not exhaustive")
    (args.out/"records.json").write_text(json.dumps(results,indent=2)+"\n")
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
