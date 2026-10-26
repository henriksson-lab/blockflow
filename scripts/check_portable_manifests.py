#!/usr/bin/env python3
"""Reject dependency paths outside the checkout, including optional dependencies."""
from pathlib import Path
import subprocess
import sys
import tomllib
import unittest


def dependency_paths(table):
    for key, value in table.items():
        if key in ('dependencies', 'dev-dependencies', 'build-dependencies'):
            for name, spec in value.items():
                if isinstance(spec, dict) and 'path' in spec:
                    yield name, spec['path']
        elif key == 'patch':
            for source in value.values():
                for name, spec in source.items():
                    if isinstance(spec, dict) and 'path' in spec:
                        yield name, spec['path']
        elif isinstance(value, dict):
            yield from dependency_paths(value)


def outside_checkout(root, manifest, dependency):
    resolved = (manifest.parent / dependency).resolve()
    return not resolved.is_relative_to(root.resolve())


def main():
    root = Path(__file__).resolve().parents[1]
    names = subprocess.check_output(
        ['git', '-C', str(root), 'ls-files', '-z', '--', 'Cargo.toml', '**/Cargo.toml']
    ).decode().split('\0')
    failures = []
    count = 0
    for name in filter(None, names):
        manifest = root / name
        if not manifest.is_file():
            continue
        table = tomllib.loads(manifest.read_text())
        for crate, path in dependency_paths(table):
            count += 1
            if outside_checkout(root, manifest, path):
                failures.append(f'{name}: {crate} = {path!r} escapes the checkout')
            elif not (manifest.parent / path / 'Cargo.toml').is_file():
                failures.append(f'{name}: {crate} = {path!r} has no Cargo.toml')
    if failures:
        print('\n'.join(failures), file=sys.stderr)
        print('Use a published revision or an in-repository crate; optional paths must resolve too.', file=sys.stderr)
        return 1
    print(f'Portable dependency paths: {count} checked')
    return 0


class PortabilityTests(unittest.TestCase):
    def test_optional_and_target_specific_dependencies_are_checked(self):
        table = tomllib.loads('''[dependencies]
optional = { path = "../outside", optional = true }
[target.'cfg(unix)'.build-dependencies]
platform = { path = "inside" }
[patch.crates-io]
patched = { path = "vendor/patched" }
[[bin]]
path = "src/main.rs"
''')
        self.assertEqual(list(dependency_paths(table)), [
            ('optional', '../outside'), ('platform', 'inside'), ('patched', 'vendor/patched')])

    def test_workspace_siblings_are_allowed_but_parent_checkout_is_not(self):
        root = Path('/tmp/portable-test-repo')
        manifest = root / 'examples/one/Cargo.toml'
        self.assertFalse(outside_checkout(root, manifest, '../..'))
        self.assertTrue(outside_checkout(root, manifest, '../../../candle'))


if __name__ == '__main__':
    if '--self-test' in sys.argv:
        unittest.main(argv=[sys.argv[0]])
    else:
        sys.exit(main())
