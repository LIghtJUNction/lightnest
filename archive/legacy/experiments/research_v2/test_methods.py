"""Small mechanism checks. Do not use model accuracy claims as unit tests."""
import unittest
from .methods import BackwardSlice, DenseEditHead, validate, evaluate_plans, stop_action, patch_action
from rgb.runtime import common
from lightnest_data.generator import generate_pair
from lightnest_data.vm import execute_b

COMMAND='change every add that contributes to the output to sub'

class MethodsTest(unittest.TestCase):
    def row(self,source,command=COMMAND):
        return {'command':command,'binary':common.assemble(source).hex()}
    def test_input_surface_rejects_answers(self):
        row=self.row('lw x1, 0(x0)\naddi x10, x1, 0\nebreak')
        with self.assertRaises(ValueError): validate(row|{'target':row['binary']})
    def test_unknown_command_rejected(self):
        with self.assertRaises(ValueError): validate(self.row('ebreak','fix all errors'))
    def test_noop_command(self):
        row=self.row('lw x1, 0(x0)\nlw x2, 4(x0)\nadd x10, x1, x2\nebreak','do not change add to sub')
        self.assertEqual(BackwardSlice().plan([row])[0],[stop_action()])
    def test_unused_add_not_selected(self):
        row=self.row('lw x1, 0(x0)\nlw x2, 4(x0)\nadd x3, x1, x2\nadd x10, x1, x2\nebreak')
        plan=BackwardSlice().plan([row])[0]
        self.assertEqual([a['position'] for a in plan if not a['stop']],[15])
    def test_dependency_chain(self):
        row=self.row('lw x1, 0(x0)\nlw x2, 4(x0)\nadd x3, x1, x2\nadd x10, x3, x2\nebreak')
        self.assertEqual([a['position'] for a in BackwardSlice().plan([row])[0] if not a['stop']],[11,15])
    def test_killed_definition(self):
        row=self.row('lw x1, 0(x0)\nlw x2, 4(x0)\nadd x3, x1, x2\naddi x3, x1, 0\naddi x10, x3, 0\nebreak')
        self.assertEqual(BackwardSlice().plan([row])[0],[stop_action()])
    def test_computed_address_rejected(self):
        with self.assertRaises(ValueError): validate(self.row('lw x1, 0(x0)\nlw x10, 0(x1)\nebreak'))
    def test_missing_stop_rejected(self):
        with self.assertRaises(ValueError): validate(self.row('lw x1, 0(x0)'))
    def test_oversize_rejected(self):
        with self.assertRaises(ValueError):validate(self.row('addi x0, x0, 0\n'*256+'ebreak'))
    def test_parameter_count(self):
        self.assertEqual(sum(p.numel() for p in DenseEditHead().parameters()),8641)
    def test_budget_is_not_reset(self):
        source='lw x1, 0(x0)\nlw x2, 4(x0)\n'+'add x1, x1, x2\n'*9+'addi x10, x1, 0\nebreak'
        row=self.row(source)
        plan=BackwardSlice().plan([row])[0]
        self.assertEqual(len(plan),10)
        target=bytearray.fromhex(row['binary'])
        for a in plan:
            if not a['stop']:target[a['position']]=a['value']
        ep=row|{'id':'budget','pair_id':'budget','lane':0,'family':'mechanism','target':target.hex(),
                'edits':[{'offset':a['position'],'value':a['value']} for a in plan if not a['stop']],
                'cases':[[[1,2],execute_b(bytes(target),[1,2])]]}
        rec=evaluate_plans(BackwardSlice(),[ep])[0]
        self.assertFalse(rec['strict']);self.assertFalse(rec['explicit_stop'])
        self.assertEqual(len(rec['actions']),9)
    def test_add_sub_graph_invariance_mechanism(self):
        for family in ('mixed','overwrite','spill','branch','loop','branch_spill','loop_spill'):
            for r in generate_pair(family,703321,0,(3,6),'mechanism'):
                self.assertEqual(common.graph(bytes.fromhex(r['binary'])),common.graph(bytes.fromhex(r['target'])))

if __name__=='__main__':unittest.main()
