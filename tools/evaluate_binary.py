"""Evaluate one complete Linux x86-64 ELF against independently supplied cases."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightnest.runtime.artifact import MAX_IMAGE, inspect_elf
from lightnest.runtime.execute import Case, ExecutableRunner


def evaluate(image: bytes, suite: dict, runner: ExecutableRunner) -> dict:
    if not image or len(image) > MAX_IMAGE:
        raise ValueError("image must contain between 1 and MAX_IMAGE bytes")
    if set(suite) != {"name", "cases"} or not isinstance(suite["name"], str):
        raise ValueError("suite requires name and cases")
    if not isinstance(suite["cases"], list) or not 1 <= len(suite["cases"]) <= 256:
        raise ValueError("suite must contain 1 to 256 cases")
    # Validate the whole suite before executing any candidate.
    cases = []
    for item in suite["cases"]:
        if set(item) != {"input", "expected"} or set(item["expected"]) != {"returncode", "stdout", "stderr"}:
            raise ValueError("each case needs input and expected returncode, stdout and stderr")
        case = Case.parse(item["input"])
        expected = item["expected"]
        if type(expected["returncode"]) is not int or not 0 <= expected["returncode"] <= 255:
            raise ValueError("expected returncode must be 0..255")
        normalized = {"returncode": expected["returncode"]}
        for key in ("stdout", "stderr"):
            normalized[key] = bytes.fromhex(expected[key]).hex()
        cases.append((case, normalized))
    try:
        info = inspect_elf(image)
        elf = {"parsed": True, "machine": info["machine"], "type": info["type"],
               "entry_in_executable_segment": info["entry_in_executable_segment"]}
    except ValueError as error:
        elf = {"parsed": False, "error": str(error)}
    results = []
    for case, expected in cases:
        actual = runner.run(image, case)
        passed = actual["status"] == "ok" and all(actual[k] == v for k, v in expected.items())
        results.append({"input": case.json(), "expected": expected, "actual": actual, "passed": passed})
    return {"suite": suite["name"], "image_sha256": hashlib.sha256(image).hexdigest(),
            "image_bytes": len(image), "runner_policy_sha256": runner.policy, "elf": elf,
            "passed_cases": sum(item["passed"] for item in results), "total_cases": len(results),
            "all_passed": all(item["passed"] for item in results), "cases": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("suite", type=Path, help="JSON cases with hex-encoded stdin/stdout/stderr")
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args()
    image = args.image.read_bytes()
    suite_bytes = args.suite.read_bytes()
    with tempfile.TemporaryDirectory(prefix="lightnest-evaluation-") as temp:
        report = evaluate(image, json.loads(suite_bytes), ExecutableRunner(Path(temp), timeout=args.timeout))
    report["suite_sha256"] = hashlib.sha256(suite_bytes).hexdigest()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
