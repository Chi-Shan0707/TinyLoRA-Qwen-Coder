"""CPU/unit checks: no model imports, downloads or experiment entrypoints."""
import ast
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def definitions(filename, names, namespace=None):
    tree = ast.parse((ROOT / filename).read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    scope = dict(namespace or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec'), scope)
    return scope


class PairConversionTests(unittest.TestCase):
    def setUp(self):
        self.convert = definitions('download_DeepCoder-Preview-Dataset.py', {'convert_tests_format'}, {'json': json})['convert_tests_format']

    def test_none_pairs_are_removed_together_and_empty_strings_preserved(self):
        tests = [{'input': str(i), 'output': f'o{i}'} for i in range(5)]
        tests += [{'input': None, 'output': 'wrong'}, {'input': 'wrong', 'output': None}, {'input': '', 'output': ''}, {}]
        expected = {'inputs': ['0', '1', '2', '3', '4', ''], 'outputs': ['o0', 'o1', 'o2', 'o3', 'o4', '']}
        self.assertEqual(self.convert(tests), expected)
        self.assertEqual(self.convert(json.dumps(tests)), expected)

    def test_dict_alignment_and_minimum_complete_pairs(self):
        self.assertEqual(self.convert({'inputs': ['a', None, 'b', 'c', 'd', 'e', 'tail'], 'outputs': ['A', 'bad', 'B', 'C', 'D', 'E']}), {'inputs': ['a', 'b', 'c', 'd', 'e'], 'outputs': ['A', 'B', 'C', 'D', 'E']})
        self.assertIsNone(self.convert([{'input': 'a', 'output': 'A'}] * 4 + [{'input': None, 'output': 'B'}]))


@unittest.skipUnless(shutil.which('g++'), 'requires existing g++')
class JudgeTests(unittest.TestCase):
    def test_exit_status_and_partial_reward(self):
        judge = definitions('utils.py', {'compile_and_run'}, {'re': re, 'subprocess': subprocess, 'tempfile': tempfile, 'os': __import__('os')})['compile_and_run']
        cases = [{'input': '', 'output': '4'}]
        self.assertEqual(judge('#include <iostream>\nint main(){std::cout<<4;return 1;}', cases), 0.5)
        self.assertEqual(judge('#include <iostream>\nint main(){std::cout<<4;return 0;}', cases), 1.0)
        self.assertEqual(judge('#include <iostream>\nint main(){std::cout<<4;}', cases + [{'input': '', 'output': '9'}]), 0.75)


class EvaluationTests(unittest.TestCase):
    def test_validation_decodes_only_completion_and_counts_actual_partial_pass(self):
        import torch
        class Model:
            device = 'cpu'
            def eval(self): pass
            def train(self): pass
            def generate(self, **kwargs): return torch.tensor([[8, 9, 10]])
        class Tokenizer:
            eos_token_id = 0
            def __call__(self, *args, **kwargs): return {'input_ids': torch.tensor([[8, 9]])}
            def decode(self, tokens, **kwargs):
                self_tokens.append(tokens.tolist())
                return 'C++'
        self_tokens = []
        rewards = iter([0.5, 0.75, 1.0])
        scope = definitions('validate.py', {'run_validation'}, {'torch': torch, 'tqdm': lambda seq, **kwargs: seq, 'convert_hf_tests_to_list': lambda tests: [tests] if tests else [], 'extract_code_from_response': lambda response: response, 'compile_and_run': lambda *args: next(rewards)})
        result = scope['run_validation'](Model(), Tokenizer(), [{'prompt': 'special prompt', 'public_tests': {'input': '', 'output': ''}}] * 3, num_samples=3)
        self.assertEqual(self_tokens, [[10], [10], [10]])
        self.assertEqual((result['zero_pass'], result['partial_pass'], result['full_pass']), (1, 1, 1))

    def test_checkpoint_rank_and_injection_seed_order(self):
        for filename in ('test.py', 'validate.py'):
            tree = ast.parse((ROOT / filename).read_text())
            injections = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'apply_tiny_lora']
            self.assertEqual(len(injections), 1)
            call = injections[0]
            self.assertTrue(any(arg.arg == 'rank' and isinstance(arg.value, ast.Name) and arg.value.id == 'rank' for arg in call.keywords))
            seeds = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'manual_seed']
            loads = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'get_model_and_tokenizer']
            self.assertTrue(any(loads[0] < line < call.lineno for line in seeds))


class StandaloneAndSaveTests(unittest.TestCase):
    def test_standalone_baseline_and_rank_four_checkpoint(self):
        import torch
        from types import SimpleNamespace
        class Dataset(list):
            column_names = ['prompt']
        class Model:
            device = 'cpu'
            model = SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(q_proj=SimpleNamespace(weight=torch.zeros(1))))])
            def eval(self): pass
            def parameters(self): return [torch.nn.Parameter(torch.zeros(4))]
            def generate(self, **kwargs): return torch.tensor([[8, 9, 10]])
        decoded, injected = [], []
        class Tokenizer:
            eos_token_id = 0
            def __call__(self, *args, **kwargs): return {'input_ids': torch.tensor([[8, 9]])}
            def decode(self, tokens, **kwargs):
                decoded.append(tokens.tolist())
                return 'C++'
        class Params:
            def __init__(self, **kwargs): self.global_v = torch.zeros(kwargs['u_dim'])
        checkpoint = {'u_value': 4, 'rank': 4, 'seed': 212, 'global_v': torch.zeros(4), 'is_quantized': False}
        samples = Dataset([{'prompt': 'special prompt', 'public_tests': {'input': '', 'output': ''}}] * 3)
        for baseline in (True, False):
            rewards = iter([0.5, 0.75, 1.0])
            scope = definitions('test.py', {'test_model'}, {
                'torch': torch, 'os': SimpleNamespace(path=SimpleNamespace(exists=lambda path: True)),
                'get_model_and_tokenizer': lambda *args, **kwargs: (Model(), Tokenizer()),
                'TinyLoRAGlobalParams': Params,
                'apply_tiny_lora': lambda *args, **kwargs: injected.append(kwargs['rank']) or 1,
                'load_dataset': lambda *args, **kwargs: samples,
                'tqdm': lambda seq, **kwargs: seq,
                'convert_hf_tests_to_list': lambda tests: [tests] if tests else [],
                'extract_code_from_response': lambda response: response,
                'compile_and_run': lambda *args: next(rewards),
            })
            with patch.object(torch, 'load', return_value=checkpoint):
                result = scope['test_model']('fake.pt', num_samples=3, baseline=baseline, use_quant=False)
            self.assertEqual((result['zero_pass'], result['partial_pass'], result['full_pass']), (1, 1, 1))
        self.assertEqual(decoded, [[10]] * 6)
        self.assertEqual(injected, [4])

    def test_callback_save_only_on_primary_process_and_final_save_guard(self):
        import torch
        import sys
        from types import SimpleNamespace, ModuleType
        fake_validate = ModuleType('validate')
        fake_validate.run_validation = lambda **kwargs: {'pass_at_1': 1.0}
        scope = definitions('train_rl.py', {'ValidationCallback'}, {
            'torch': torch, 'os': __import__('os'), 'TrainerCallback': object, 'USE_QUANT': False,
        })
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'validate': fake_validate}):
            for primary in (False, True):
                callback = scope['ValidationCallback']([], 1, 1, tmp, SimpleNamespace(global_v=torch.zeros(4)), 4, 4, 212, 'fake', 1)
                state = SimpleNamespace(global_step=1, is_world_process_zero=primary)
                with patch.object(torch, 'save') as save:
                    callback.on_step_end(None, state, None, model=object(), processing_class=object())
                self.assertEqual(save.call_count, int(primary))
        tree = ast.parse((ROOT / 'train_rl.py').read_text())
        guarded = [node for node in tree.body if isinstance(node, ast.If)
                   and isinstance(node.test, ast.Call) and isinstance(node.test.func, ast.Attribute)
                   and node.test.func.attr == 'is_world_process_zero']
        self.assertEqual(len(guarded), 1)
        self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and node.func.attr == 'save' for node in ast.walk(guarded[0])))


class TinyLayerTests(unittest.TestCase):
    def test_shared_gradient_and_svd_failure(self):
        import torch
        scope = definitions('utils.py', {'TinyLoRAGlobalParams', 'TinyLoRALinear'}, {'torch': torch, 'nn': torch.nn, 'F': torch.nn.functional})
        params = scope['TinyLoRAGlobalParams'](u_dim=3)
        layer_type = scope['TinyLoRALinear']
        first = layer_type(torch.nn.Linear(4, 4, bias=False), rank=2, global_params_ref=params)
        second = layer_type(torch.nn.Linear(4, 4, bias=False), rank=2, global_params_ref=params)
        self.assertIs(first.global_params_ref.global_v, second.global_params_ref.global_v)
        self.assertEqual(tuple(first.P.shape), (3, 2, 2))
        self.assertTrue(torch.equal(first.P, second.P))
        # Verify the upstream low-rank forward against its full-matrix equation.
        with torch.no_grad():
            params.global_v.copy_(torch.tensor([0.1, -0.2, 0.3]))
        x = torch.randn(2, 4)
        mixing = torch.einsum('u,urs->rs', params.global_v.float(), first.P.float())
        delta = first.U.float() @ torch.diag(first.S.float()) @ mixing @ first.Vh.float()
        self.assertTrue(torch.allclose(first._compute_delta_output(x, torch.float32), x @ delta.T, atol=1e-6))
        first(torch.randn(2, 4)).float().sum().backward()
        self.assertTrue(torch.isfinite(params.global_v.grad).all())
        self.assertGreater(params.global_v.grad.abs().sum().item(), 0)
        with patch.object(torch.linalg, 'svd', side_effect=RuntimeError('forced failure')):
            with self.assertRaisesRegex(RuntimeError, 'initialization failed'):
                layer_type(torch.nn.Linear(4, 4), rank=2, global_params_ref=params)


if __name__ == '__main__':
    unittest.main()
