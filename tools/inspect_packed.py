"""Inspect an embedded code archive without loading the executable or language weights."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightnest_rgb.packed import PackedArtifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('--module')
    choice.add_argument('--code-entry')
    parser.add_argument('--module-output', type=Path)
    args = parser.parse_args()
    selected = args.module or args.code_entry
    if selected and not args.module_output:
        parser.error('code extraction requires --module-output')
    with PackedArtifact(args.image) as artifact:
        report = artifact.metadata()
        if selected:
            raw = artifact.read_module(selected) if args.module else artifact.read_code_member(selected)
            with args.module_output.open('xb') as file:
                file.write(raw)
            report['extracted'] = {'module': selected, 'container': 'PYZ' if args.module else 'CArchive', 'bytes': len(raw),
                                   'sha256': hashlib.sha256(raw).hexdigest(),
                                   'path': str(args.module_output), 'sanitized': False}
        report['bytes_read'] = artifact.bytes_read
    with args.output.open('x') as file:
        json.dump(report, file, indent=2); file.write('\n')
    print(json.dumps({key: value for key, value in report.items() if key not in {'members', 'modules'}}, indent=2))
    print(json.dumps({'members': len(report['members']), 'modules': len(report['modules'])}))


if __name__ == '__main__':
    main()
