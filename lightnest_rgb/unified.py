"""One artifact agent for complete generation and executable editing."""
from __future__ import annotations

from pathlib import Path
import asyncio

from .artifact import inspect_elf, write_image
from .execute import Case, ExecutableRunner
from .iterate import Example, RGBEditor
from .training import load_generator, load_policy
from .self_agent import SelfAgent


class RGBArtifactAgent:
    """Unified model-facing API for making or changing an ELF artifact."""
    def __init__(self, policy_checkpoint=None, generator_checkpoint=None, *, workdir):
        self.policy = load_policy(policy_checkpoint)[0] if policy_checkpoint else None
        self.generator = load_generator(generator_checkpoint)[0] if generator_checkpoint else None
        self.runner = ExecutableRunner(Path(workdir) / "execution", cache=False)

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

    def inspect_project(self, root, structure_checkpoint=None):
        """Run the generic async inspection tools against an unseen project."""
        return asyncio.run(SelfAgent().inspect(Path(root), structure_checkpoint))
