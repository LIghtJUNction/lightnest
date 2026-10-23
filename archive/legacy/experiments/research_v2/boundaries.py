"""Semantic counterexamples, separate from the structural-label benchmark.

The original task labels structural dependence. These probes expose why 100%
in that task would NOT establish semantic program understanding.
"""
import argparse
import json
from pathlib import Path
from .methods import BackwardSlice,common
from .prepare import write
from lightnest_data.vm import execute_a,execute_b


def main(root):
    command='change every add that contributes to the output to sub'
    examples={
      'self_xor':'lw x1, 0(x0)\nlw x2, 4(x0)\nadd x3, x1, x2\nxor x10, x3, x3\nebreak',
      'self_sub':'lw x1, 0(x0)\nlw x2, 4(x0)\nadd x3, x1, x2\nsub x10, x3, x3\nebreak',
      'multiply_zero':'lw x1, 0(x0)\nlw x2, 4(x0)\nadd x3, x1, x2\nmul x10, x3, x0\nebreak',
      'and_zero':'lw x1, 0(x0)\nlw x2, 4(x0)\nadd x3, x1, x2\nand x10, x3, x0\nebreak',
      'infeasible_path':'lw x1, 0(x0)\nlw x2, 4(x0)\nbeq x1, x1, yes\nadd x3, x1, x2\naddi x10, x3, 0\njal x0, done\nyes:\naddi x10, x0, 7\ndone:\nebreak',
    }
    inputs=[[0,0],[1,2],[-1,3],[2**31-1,-2**31],[7,7],[-99,15],[64,32],[-2**31,-1]]
    results=[]
    for name,source in examples.items():
        original=common.assemble(source)
        plan=BackwardSlice().plan([{'command':command,'binary':original.hex()}])[0]
        revised=bytearray(original)
        for act in plan:
            if not act['stop']:revised[act['position']]=act['value']
        outputs=[]
        for inp in inputs:
            values=[fn(code,inp) for fn in (execute_a,execute_b) for code in (original,bytes(revised))]
            assert len(set(values))==1
            outputs.append({'inputs':inp,'four_vm_outputs':values})
        assert any(not a['stop'] for a in plan)
        results.append({'name':name,'source':source,'original':original.hex(),'candidate':bytes(revised).hex(),
           'actions':plan,'finite_input_checks':outputs,'structural_edit_proposed':True,
           'semantic_influence':False,
           'analytic_reason':'x XOR x / x-x / x*0 / x&0 is always zero, or branch x==x makes the edited arm unreachable'})
    write(root/'semantic-boundaries.json',{'separate_from_original_benchmark':True,'examples':results,
         'conclusion':'Static dependency reachability is not exact semantic influence; no general repair or intelligence claim.'})
    print('BOUNDARIES',len(results),'analytical semantic counterexamples confirmed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
