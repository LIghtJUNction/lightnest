"""Create a grouped new holdout without querying any model."""
import argparse
import gzip
import hashlib
import json
import random
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from . import BASE, ARTIFACT_ROOT
from lightnest_data.core import neutral_fingerprint
from lightnest_data.generator import generate_pair, FAMILIES
from lightnest_data.vm import execute_a, execute_b
from study.prepare import simple_pair


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return [json.loads(line) for line in Path(p).read_text().splitlines()]


def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def main(output, index_archive):
    output.mkdir(parents=True, exist_ok=False)
    known, exact = set(), set()
    index_rows = 0
    with zipfile.ZipFile(index_archive) as archive:
        for name in archive.namelist():
            if name.endswith('.jsonl.gz') and 'index' in name:
                with gzip.GzipFile(fileobj=archive.open(name)) as stream:
                    for line in stream:
                        r = json.loads(line)
                        known.add(r['neutral_sha256'])
                        exact.update((r['binary_sha256'], r['target_sha256']))
                        index_rows += 1
    historical_paths = [*ARTIFACT_ROOT.glob('data/*.jsonl'), *BASE.glob('data/*.jsonl'),
                        *BASE.glob('baseline_data/*.jsonl')]
    for path in historical_paths:
        for row in read(path):
            for k in ('binary', 'target'):
                b = bytes.fromhex(row[k])
                known.add(neutral_fingerprint(b))
                exact.add(hashlib.sha256(b).hexdigest())
    families = ('simple',) + FAMILIES + ('branch_spill', 'loop_spill')
    rng = random.Random(20261020201)
    inputs = [[0,0,0],[1,2,3],[-1,2,-3],[2**31-1,1,-1],[-2**31,-1,1],[7,7,0],[-13,13,5],[2,5,3]]
    inputs += [[rng.randint(-100000,100000) for _ in range(3)] for _ in range(8)]
    rows = []
    collisions = calls = 0
    for fi, family in enumerate(families):
        count = attempt = 0
        while count < 24:
            seed = 20261020200 + 100000*fi + attempt
            attempt += 1
            bounds = (3,8) if count < 12 else (13,20)
            batch = (simple_pair(seed, count%2, 'research-v2-holdout') if family == 'simple' else
                     generate_pair(family, seed, count%2, bounds, 'research-v2-holdout'))
            fs = {neutral_fingerprint(bytes.fromhex(r[k])) for r in batch for k in ('binary','target')}
            xs = {hashlib.sha256(bytes.fromhex(r[k])).hexdigest() for r in batch for k in ('binary','target')}
            if fs & known or xs & exact:
                collisions += 1
                continue
            known.update(fs)
            exact.update(xs)
            for r in batch:
                r.pop('source', None)
                r['difficulty'] = 'normal' if count < 12 else ('simple_control' if family == 'simple' else 'long_13_20')
                r['cases'] = []
                for inp in inputs:
                    for field in ('binary','target'):
                        a = execute_a(bytes.fromhex(r[field]), inp)
                        b = execute_b(bytes.fromhex(r[field]), inp)
                        calls += 2
                        assert a == b
                        if field == 'target':
                            r['cases'].append([inp,a])
            rows.extend(batch)
            count += 1
    with (output/'holdout.jsonl').open('x') as f:
        for row in rows:
            f.write(json.dumps(row, separators=(',',':'))+'\n')
    protocol = {
        'id':'lightnest-research-v2','created_utc':datetime.now(timezone.utc).isoformat(),
        'source_commit':'0b3d1ffb98be169563fb4aef69c995122cb80726',
        'seeds':[104729,130363],
        'methods':['C64','stable_exit','dense16','dense64','backward_slice'],
        'training':{'pool':'Hybrid v1 fit TRAIN only','steps':768,'batch_states':32,'lr':0.001,
                    'frozen':'entire inherited C64','trainable':'133->64->1 edit classifier, 8641 parameters',
                    'loss':'per-state balanced positive/negative BCE; no test labels or test selection',
                    'positive_weight':2,'feature_depths':[16,64]},
        'calibration':{'source':'352 previously public calibration rows','dense_thresholds':[-2,-1,0,1,2,3,4,6],
                       'stable_thresholds':[0.9,0.99,0.999,1.01],
                       'dense_selection':'min false_edits, then max strict_edits, then max strict_total, then smallest absolute threshold',
                       'early_selection':'zero lost baseline success and zero added false edit, then fewest rounds'},
        'selection':{'quality_gate':'no new false edits and no lost C64 success on public regression; zero holdout false edits required',
                     'development_ranking':'feasible first, then total strict success; tie use measured runtime',
                     'holdout_rule':'report all frozen endpoints once; do not retune or refit after holdout; do not deploy automatically'},
        'evaluation':{'full_public_regression':2372,'new_holdout_rows':len(rows),'new_holdout_groups':192,
                      'new_edits':384,'new_noops':384,'max_edits':8,'explicit_STOP':True,'batch':16,
                      'new_models_queried_during_preparation':0,'vm_inputs_per_task':16},
        'scope':['existing synthetic families; fresh seeds and longer 13..20 chains, not real defects',
                 'symbolic and dense methods get a documented 12-command grammar and typed ADD/SUB patch prior',
                 'backward slicing is static analysis, not trained neural intelligence',
                 'no external model API, GPU, paid compute or native candidate execution'],
        'inputs':{'fit':sha(ARTIFACT_ROOT/'data/fit.jsonl'),'calibration':sha(BASE/'baseline_data/calibration.jsonl'),
                  'regression':sha(BASE/'baseline_data/regression.jsonl'),'holdout':sha(output/'holdout.jsonl'),
                  'full_index_archive':sha(index_archive)},
        'baseline_weights':{str(s):sha(BASE/f'models/rgb-{s}.pt') for s in (104729,130363)},
    }
    write(output/'protocol.json', protocol)
    write(output/'construction.json', {'indexed_historical_rows':index_rows,
          'historical_sources':{str(p.relative_to(ARTIFACT_ROOT)):sha(p) for p in historical_paths},
          'holdout_sha256':sha(output/'holdout.jsonl'),'families':dict(Counter(r['family'] for r in rows)),
          'rejected_collisions':collisions,'two_vm_construction_calls':calls,
          'isolated_from_current_splits_and_10k_index':True,'full_semantic_deduplication':False})
    print('FROZEN',len(rows),'fresh records;',index_rows,'historical index rows;',calls,'VM construction calls',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--index-archive',type=Path,required=True)
    a=parser.parse_args()
    main(a.output,a.index_archive)
