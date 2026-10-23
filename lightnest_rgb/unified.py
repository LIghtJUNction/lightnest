"""One artifact agent for complete generation and executable editing."""
from __future__ import annotations

from pathlib import Path
import asyncio
import torch

from .artifact import inspect_elf, write_image
from .any2any import ArtifactCore, ArtifactRequest, KINDS, apply_copy_paste
from .execute import Case, ExecutableRunner
from .iterate import Example, RGBEditor
from .training import load_generator, load_policy
from .self_agent import SelfAgent


class RGBArtifactAgent:
    """Unified model-facing API for understanding, editing and making artifacts."""
    def __init__(self, policy_checkpoint=None, generator_checkpoint=None, any2any_checkpoint=None, *, workdir):
        self.policy = load_policy(policy_checkpoint)[0] if policy_checkpoint else None
        self.generator = load_generator(generator_checkpoint)[0] if generator_checkpoint else None
        self.any2any = None
        if any2any_checkpoint:
            payload=torch.load(any2any_checkpoint,map_location='cpu',weights_only=True)
            if payload.get('architecture') != 'lightnest-any2any-v1' or payload.get('kind') != 'artifact-core':
                raise ValueError('checkpoint is not an Any2Any artifact core')
            self.any2any=ArtifactCore(**payload['config'])
            missing,unexpected=self.any2any.load_state_dict(payload['state_dict'],strict=False)
            allowed={name for name in missing if name.startswith(('copy_query.','budget.'))}
            if set(unexpected) or set(missing) != allowed:
                raise ValueError(f'Any2Any checkpoint mismatch: {missing} {unexpected}')
            self.any2any.eval()
        self.runner = ExecutableRunner(Path(workdir) / "execution", cache=False)

    def understand(self, raw, request):
        if self.any2any is None: raise ValueError("Any2Any checkpoint is required")
        if not isinstance(request, ArtifactRequest): raise TypeError("request must be an ArtifactRequest")
        with torch.inference_mode():
            abstract, heads = self.any2any.summarize(raw, request)
            kind_index=int(heads['kind_logits'].argmax(-1)[0])
            budget=heads['budget']
            return {'kind':request.kind,'predicted_kind':KINDS[kind_index] if kind_index < len(KINDS) else 'custom',
                    'summary':heads['summary'][0].tolist(),'budget':{'index':budget['index'],'depth':budget['depth']},
                    'abstract':abstract[0].tolist()}

    def copy_paste(self, raw, request, expected_sha256, *, output=None, apply=False):
        if self.any2any is None: raise ValueError("Any2Any checkpoint is required")
        if not isinstance(request, ArtifactRequest): raise TypeError("request must be an ArtifactRequest")
        action, evidence=self.any2any.propose_copy_paste(raw, request)
        result={'action':action.__dict__,'budget':evidence['budget'],'source_sha256':expected_sha256}
        if apply:
            changed=apply_copy_paste(raw, action, expected_sha256)
            if output is not None: result['artifact']=write_image(Path(output), changed)
            result['sha256']=__import__('hashlib').sha256(changed).hexdigest()
        return result

    def generate(self, request, *, output=None):
        if self.generator is None: raise ValueError("generator checkpoint is required")
        result = self.generator.generate(request); raw = result.pop("bytes")
        if not result["stop"]: raise ValueError("generator did not stop")
        info = inspect_elf(raw); actual = self.runner.run(raw, Case())
        if (actual["status"] != "ok" or actual["returncode"] != request.get("exit", 0)
                or actual["stdout"] != request.get("stdout", "") or actual["stderr"]):
            raise ValueError("generated ELF failed requested behavior")
        if output is not None: result["artifact"] = write_image(Path(output), raw)
        result.update({"elf": info, "execution": actual}); return result

    def repair(self, image, command, examples, *, output=None, mode="routed", fixed_depth=None):
        if self.policy is None: raise ValueError("policy checkpoint is required")
        result = RGBEditor(self.policy, self.runner).repair(image, command, examples, mode=mode, fixed_depth=fixed_depth)
        raw = result.pop("bytes")
        if not result["stop"]: raise ValueError("editor abstained")
        if output is not None: result["artifact"] = write_image(Path(output), raw)
        result["elf"] = inspect_elf(raw); return result

    def inspect_project(self, root, structure_checkpoint=None, any2any_checkpoint=None):
        """Run the generic async inspection tools against an unseen project."""
        return asyncio.run(SelfAgent().inspect(Path(root), structure_checkpoint)) if not any2any_checkpoint else asyncio.run(__import__('lightnest_rgb.semantic_loop',fromlist=['SemanticLoop']).SemanticLoop().inspect(Path(root), any2any_checkpoint))
