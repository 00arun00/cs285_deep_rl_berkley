#!/usr/bin/env python3
"""Reconcile native API output with source and enforce production-source documentation completeness."""
import argparse
import ast
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
from collections import Counter
import zlib

import griffe

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'hw1/src'
PILOT = 'hw1_imitation.data.Normalizer'


def walk_classes(body, prefix, local=False):
    for node in body:
        if isinstance(node, ast.ClassDef):
            yield prefix + '.' + node.name, node, local
            yield from walk_classes(node.body, prefix + '.' + node.name, local)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield from walk_classes(node.body, prefix + '.' + node.name, True)
        else:
            for _, value in ast.iter_fields(node):
                if isinstance(value, list):
                    yield from walk_classes([v for v in value if isinstance(v, ast.AST)], prefix, local)


def assigned_names(target, instance=False):
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(assigned_names(t, instance) for t in target.elts))
    if instance and isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
        return {target.attr}
    if not instance and isinstance(target, ast.Name):
        return {target.id}
    return set()


def own_nodes(body):
    for node in body:
        yield node
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        for child in ast.iter_child_nodes(node):
            yield from own_nodes([child])


def parameters(node, *, bound=False):
    positional = [*node.args.posonlyargs, *node.args.args]
    if bound and positional and not any(ast.unparse(d) in {'staticmethod', 'builtins.staticmethod'} for d in node.decorator_list):
        positional = positional[1:]
    args = [*positional, *node.args.kwonlyargs]
    args += [a for a in (node.args.vararg, node.args.kwarg) if a]
    return {a.arg for a in args}


def documented(obj, kind):
    result = {}
    if obj.docstring:
        for section in obj.docstring.parsed:
            if section.kind.value == kind:
                result.update({e.name.lstrip('*'): e.description for e in section.value})
    return result


def resolve_own_member(class_object, method):
    """Match an AST method to its own Griffe docs, including property accessors.

    A property is stored as one Attribute with the getter documentation, plus
    optional setter/deleter Functions. Never substitute getter docs when an
    accessor cannot be matched. Source spans also reject inherited or aliased
    metadata that does not describe this explicit method definition.
    """
    accessor = next((decorator.attr for decorator in method.decorator_list
                     if isinstance(decorator, ast.Attribute)
                     and decorator.attr in {'getter', 'setter', 'deleter'}), None)
    is_property = any(ast.unparse(decorator) in {'property', 'builtins.property'}
                      for decorator in method.decorator_list)
    label = method.name + ('.'+accessor if accessor else '.getter' if is_property else '')
    member = class_object.members.get(method.name)
    if member is not None and not member.is_alias and accessor in {'setter', 'deleter'}:
        member = getattr(member, accessor, None)
    start = min([method.lineno, *[decorator.lineno for decorator in method.decorator_list]])
    if member is None or member.is_alias or member.lineno not in range(start, method.end_lineno + 1):
        return None, label, 'Cannot resolve own member documentation: '+label
    return member, label, None


def output_doc_issues(method, member):
    """Check documented outputs from annotations and the method's own syntax.

    This is deliberately not runtime inference: any explicit non-None return
    expression needs a description even if it might evaluate to None. Nested
    scopes are excluded. Generator yield values and terminal return values are
    separate contracts; a generator annotation alone does not require Returns.
    """
    issues, ambiguities = [], []
    nodes = list(own_nodes(method.body))
    value_returns = [node for node in nodes if isinstance(node, ast.Return)
                     and node.value is not None
                     and not (isinstance(node.value, ast.Constant) and node.value.value is None)]
    generator = any(isinstance(node, (ast.Yield, ast.YieldFrom)) for node in nodes)
    # Read the AST decorator, not the shared Griffe property labels: a setter
    # has the same member name but does not inherit the getter's output contract.
    property_getter = any(ast.unparse(decorator) in {'property', 'builtins.property'}
                          for decorator in method.decorator_list)
    annotation = ast.unparse(method.returns) if method.returns is not None else ''
    annotated_result = bool(annotation and annotation not in {'None', "'None'", '"None"'})
    returns = []
    yields = []
    if member.docstring:
        for section in member.docstring.parsed:
            if section.kind.value == 'returns': returns.extend(section.value)
            if section.kind.value == 'yields': yields.extend(section.value)
    needs_return = bool(value_returns or property_getter or (annotated_result and not generator))
    if needs_return and not returns: issues.append('Missing return description: '+method.name)
    if generator and not yields: issues.append('Missing yield description: '+method.name)
    if any(not entry.description.strip() for entry in returns): issues.append('Empty return description: '+method.name)
    if any(not entry.description.strip() for entry in yields): issues.append('Empty yield description: '+method.name)
    if annotated_result and not generator and not any(word in annotation for word in ('tuple', 'Tuple', 'Generator', 'Iterator', 'Iterable')) and len(returns)>1:
        issues.append('Multiple parsed returns for single-result annotation: '+method.name)
    if value_returns and any(not isinstance(node.value, ast.Constant) for node in value_returns):
        ambiguities.append('Return expression may evaluate to None; output documentation required without runtime inference')
    if property_getter and not value_returns:
        ambiguities.append('Property output contract requires review: no explicit value-return path was found')
    if generator and value_returns:
        ambiguities.append('Generator terminal return is separate from yielded values; review its StopIteration.value contract')
    if not generator and any(word in annotation for word in ('Generator', 'Iterator', 'Iterable')):
        ambiguities.append('Iterator annotation without local yield: Returns describes the returned object; iteration semantics require review')
    return issues, ambiguities


def callable_issues(node, obj, *, bound=False):
    issues, ambiguities = output_doc_issues(node, obj)
    if not (ast.get_docstring(node) or '').strip():
        issues.append('Missing callable docstring: '+node.name)
    params = documented(obj, 'parameters')
    for name in sorted(parameters(node, bound=bound)-params.keys()):
        issues.append('Missing parameter: '+node.name+'.'+name)
    for name in sorted(params.keys()-parameters(node, bound=bound)):
        issues.append('Stale parameter: '+node.name+'.'+name)
    for name, description in params.items():
        if not description.strip(): issues.append('Empty parameter: '+node.name+'.'+name)
    declared_raises = set()
    if obj.docstring:
        for section in obj.docstring.parsed:
            if section.kind.value == 'raises':
                for entry in section.value:
                    declared_raises.add(str(entry.annotation).rsplit('.', 1)[-1])
                    if not entry.description.strip(): issues.append('Empty raise description: '+node.name)
    # Conservatively require only explicit named raises outside try blocks.
    # Caught/rethrown/dynamically constructed exceptions need human review.
    def raises_in(body, protected=False):
        for item in body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            if isinstance(item, ast.Raise):
                exc = item.exc.func if isinstance(item.exc, ast.Call) else item.exc
                if protected or not isinstance(exc, (ast.Name, ast.Attribute)) or (not isinstance(item.exc, ast.Call) and ast.unparse(exc).rsplit('.', 1)[-1] not in {'Exception', 'BaseException', 'ValueError', 'TypeError', 'RuntimeError', 'KeyError', 'IndexError', 'OSError', 'NotImplementedError', 'AssertionError', 'SystemExit', 'KeyboardInterrupt', 'StopIteration', 'GeneratorExit'}):
                    ambiguities.append('Caught, rethrown, or dynamic raise requires semantic review')
                else:
                    name = ast.unparse(exc).rsplit('.', 1)[-1]
                    if name not in declared_raises: issues.append('Missing raise description: '+node.name+'.'+name)
            for child in ast.iter_child_nodes(item):
                raises_in([child], protected or isinstance(item, (ast.Try, ast.TryStar)))
    raises_in(node.body)
    return issues, ambiguities


def walk_functions(body, prefix, local=False, in_class=False):
    """Yield explicit functions, marking runtime-local definitions separately."""
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            path = prefix+'.'+node.name
            if local or not in_class: yield path, node, local, in_class
            yield from walk_functions(node.body, path, True)
        elif isinstance(node, ast.ClassDef):
            yield from walk_functions(node.body, prefix+'.'+node.name, local, True)
        else:
            for _, value in ast.iter_fields(node):
                if isinstance(value, list):
                    yield from walk_functions([v for v in value if isinstance(v, ast.AST)], prefix, local, in_class)


def collected_objects(obj, kind):
    for child in obj.members.values():
        if child.is_alias: continue
        if kind == 'class' and child.is_class: yield child.path
        if kind == 'function' and child.is_function and obj.is_module: yield child.path
        if child.is_module or (kind == 'class' and child.is_class):
            yield from collected_objects(child, kind)


def difference(expected, actual):
    counts = Counter(actual)
    return {'missing': sorted(expected-counts.keys()), 'extra': sorted(counts.keys()-expected),
            'duplicates': sorted(name for name, count in counts.items() if count > 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument('--all', action='store_true', help='Enforce all production modules/classes/functions (default; used by CI)')
    scope.add_argument('--pilot', help='Enforce one class for focused authoring; other content remains report-only')
    parser.add_argument('--report', type=Path, default=Path('/tmp/docs-coverage.json'))
    parser.add_argument('--baseline', type=Path, help='Optional pre-edit module comparison; requires --pilot')
    parser.add_argument('--ruff', default=os.environ.get('RUFF_BIN', shutil.which('ruff')))
    args = parser.parse_args()
    if args.baseline and not args.pilot: parser.error('--baseline requires --pilot')
    report = {'scope': 'pilot' if args.pilot else 'all', 'pilot': args.pilot,
              'modules': [], 'classes': [], 'functions': [], 'exceptions': [], 'errors': [], 'limits': [
        'Static assignments cannot establish arbitrary runtime-created attributes.',
        'Semantic accuracy requires source and rendered review; structural checks do not prove it.',
        'Inherited and imported members are excluded from authored coverage.',
        'Only explicit named raises outside try blocks are required; caught/dynamic and dependency exceptions require review.',
        'Function-local helpers are checked in source but have no independently rendered native API object.',
    ]}
    def enforce(path, issues, selected=None):
        if not args.pilot or selected == args.pilot:
            report['errors'].extend(issues if args.pilot else [path+': '+issue for issue in issues])
    try:
        package = griffe.load('hw1_imitation', search_paths=[str(SOURCE)], allow_inspection=False, docstring_parser='google')
        source_files = sorted(SOURCE.rglob('*.py'))
        if not source_files: raise ValueError('No production Python source files found')
        native = Counter()
        native_classes, native_functions = [], []
        for page in (ROOT/'site/api').rglob('index.html'):
            html = page.read_text()
            native.update(re.findall(r'id="(hw1_imitation(?:\.[^"]+)?)"', html))
            native_classes.extend(re.findall(r'<div class="doc doc-object doc-class">.*?<h[1-6][^>]*id="([^"]+)"', html, re.S))
            native_functions.extend(re.findall(r'<div class="doc doc-object doc-function">.*?<h[1-6][^>]*id="([^"]+)"', html, re.S))
        if not native: raise ValueError('No native API anchors found; build the site first')
        inventory_data = (ROOT/'site/objects.inv').read_bytes()
        inventory = Counter(line.split()[0] for line in zlib.decompress(inventory_data.split(b'\n', 4)[4]).decode().splitlines()
                            if len(line.split()) > 1 and line.split()[1].startswith('py:'))
        pilot_node = pilot_file = None
        def anchor_issues(paths):
            issues = []
            for path in sorted(paths):
                if native[path] == 0: issues.append('Missing rendered anchor: '+path)
                elif native[path] != 1: issues.append('Duplicate rendered anchor: '+path)
                if inventory[path] == 0: issues.append('Missing inventory entry: '+path)
                elif inventory[path] != 1: issues.append('Duplicate inventory entry: '+path)
            return issues
        for file in source_files:
            parts = list(file.relative_to(SOURCE).with_suffix('').parts)
            if parts[-1] == '__init__': parts.pop()
            module = '.'.join(parts)
            source = file.read_text()
            tree = ast.parse(source)
            module_obj = package if module == 'hw1_imitation' else package[module.removeprefix('hw1_imitation.')]
            module_issues = []
            if not (ast.get_docstring(tree) or '').strip(): module_issues.append('Missing module docstring')
            module_issues += anchor_issues({module})
            report['modules'].append({'path': module, 'issues': module_issues})
            enforce(module, module_issues)
            for path, node, local in walk_classes(tree.body, module):
                if local:
                    report['exceptions'].append({'path': path, 'reason': 'function-local class requires source review; native object unavailable'})
                    if not (ast.get_docstring(node) or '').strip(): enforce(path, ['Missing local class docstring'])
                    continue
                obj = package[path.removeprefix('hw1_imitation.')]
                if obj.is_alias or not obj.is_class: raise ValueError('Class extraction did not resolve own class: '+path)
                methods = [n for n in own_nodes(node.body) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                fields = set()
                for n in own_nodes(node.body):
                    if isinstance(n, (ast.Assign, ast.AnnAssign)):
                        for target in n.targets if isinstance(n, ast.Assign) else [n.target]: fields.update(assigned_names(target))
                for method in methods:
                    for n in own_nodes(method.body):
                        if isinstance(n, (ast.Assign, ast.AnnAssign)):
                            for target in n.targets if isinstance(n, ast.Assign) else [n.target]: fields.update(assigned_names(target, True))
                        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {'setattr', 'exec'}:
                            report['exceptions'].append({'path': path+'.'+method.name, 'reason': 'dynamic assignment requires review'})
                issues = []
                if not (ast.get_docstring(node) or '').strip(): issues.append('Missing class docstring')
                attrs = documented(obj, 'attributes')
                for name in fields:
                    if name in obj.members and obj[name].docstring and obj[name].docstring.value.strip(): attrs.setdefault(name, obj[name].docstring.value)
                for name in sorted(fields-attrs.keys()): issues.append('Missing attribute description: '+name)
                property_names = {n.name for n in methods if any(ast.unparse(d) in {'property', 'builtins.property'} for d in n.decorator_list)}
                for name in sorted(attrs.keys()-fields-property_names): issues.append('Stale attribute: '+name)
                for name, desc in attrs.items():
                    if not desc.strip(): issues.append('Empty attribute description: '+name)
                labels = []
                for method in methods:
                    member, label, resolution_issue = resolve_own_member(obj, method)
                    labels.append(label)
                    if not (ast.get_docstring(method) or '').strip(): issues.append('Missing member docstring: '+label)
                    if resolution_issue:
                        issues.append(resolution_issue)
                        continue
                    method_issues, ambiguities = callable_issues(method, member, bound=True)
                    issues.extend(issue.replace(': '+method.name, ': '+label) for issue in method_issues)
                    report['exceptions'].extend({'path': path+'.'+label, 'reason': reason} for reason in ambiguities)
                expected = {path, *[path+'.'+n.name for n in methods], *[path+'.'+name for name in fields]}
                anchors = anchor_issues(expected)
                # Accessor sections have native HTML headings but intentionally
                # no separate Python-object inventory registration.
                for method in methods:
                    accessor = next((d.attr for d in method.decorator_list if isinstance(d, ast.Attribute) and d.attr in {'setter', 'deleter'}), None)
                    if accessor:
                        heading = path+'.'+method.name+'--'+accessor
                        if native[heading] == 0: anchors.append('Missing rendered accessor anchor: '+heading)
                        elif native[heading] != 1: anchors.append('Duplicate rendered accessor anchor: '+heading)
                report['classes'].append({'path': path, 'file': str(file.relative_to(ROOT)), 'members': labels, 'attributes': sorted(fields), 'issues': issues,
                                          'missing_rendered_anchors': sorted(expected-native.keys()), 'rendering_issues': anchors})
                enforce(path, issues+anchors, path)
                if path == args.pilot: pilot_node, pilot_file = node, file
            for path, node, local, bound in walk_functions(tree.body, module):
                if local:
                    # Parse a source-only helper in isolation: no invented native object/anchor.
                    synthetic = griffe.visit('_source_local', filepath=file, code=ast.unparse(node), docstring_parser=griffe.Parser.google)
                    obj = synthetic[node.name]
                    report['exceptions'].append({'path': path, 'reason': 'function-local helper: source documentation checked; native discovery/anchor not available'})
                else:
                    obj = module_obj[node.name]
                    if obj.is_alias or not obj.is_function: raise ValueError('Function extraction did not resolve own function: '+path)
                    start = min([node.lineno, *[d.lineno for d in node.decorator_list]])
                    if obj.lineno not in range(start, node.end_lineno+1): raise ValueError('Function extraction source mismatch: '+path)
                issues, ambiguities = callable_issues(node, obj, bound=bound)
                report['exceptions'].extend({'path': path, 'reason': reason} for reason in ambiguities)
                anchors = [] if local else anchor_issues({path})
                report['functions'].append({'path': path, 'file': str(file.relative_to(ROOT)), 'local': local, 'issues': issues, 'rendering_issues': anchors})
                enforce(path, issues+anchors)
        ast_classes = {r['path'] for r in report['classes']}
        ast_functions = {r['path'] for r in report['functions'] if not r['local']}
        # Nested class methods are not standalone functions. Unknown module-level
        # headings remain in the inventory so unexpected output fails closed.
        standalone_native = [path for path in native_functions if not any(path.startswith(cls+'.') for cls in ast_classes)]
        for key, expected, actual in [('native_class_difference', ast_classes, native_classes),
                                      ('native_function_difference', ast_functions, standalone_native),
                                      ('extraction_difference', ast_classes, list(collected_objects(package, 'class'))),
                                      ('function_extraction_difference', ast_functions, list(collected_objects(package, 'function')))]:
            report[key] = difference(expected, actual)
            if any(report[key].values()): report['errors'].append(key+': inventory differs from source')
        if args.pilot and pilot_node is None: report['errors'].append('Pilot class not found in source inventory')
        if args.baseline and pilot_node is not None:
            def executable(path):
                tree = ast.parse(path.read_text())
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and ast.get_docstring(node): node.body = node.body[1:]
                return ast.dump(tree, include_attributes=False)
            report['executable_ast_unchanged'] = executable(args.baseline) == executable(pilot_file)
            if not report['executable_ast_unchanged']: report['errors'].append('Executable AST changed')
        if not args.ruff: report['errors'].append('Ruff not found: set RUFF_BIN or pass --ruff; no environment is installed automatically')
        else:
            with tempfile.TemporaryDirectory() as directory:
                if args.pilot and pilot_node is not None:
                    target = Path(directory)/'pilot.py'
                    target.write_text(ast.get_source_segment(pilot_file.read_text(), pilot_node)+'\n')
                    targets = [str(target)]
                else: targets = [str(file) for file in source_files]
                result = subprocess.run([args.ruff, 'check', '--config', str(ROOT/'docs/ruff.toml'), *targets], text=True, capture_output=True)
                report['ruff'] = {'passed': result.returncode == 0, 'output': result.stdout+result.stderr, 'scope': report['scope']}
                if result.returncode: report['errors'].append('Ruff docstring checks failed')
    except Exception as error:
        report['errors'].append('Documentation extraction/validation failed: '+type(error).__name__+': '+str(error))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(f"Discovered {len(report['classes'])} classes and {len(report['functions'])} functions; scope: {report['scope']}; errors: {len(report['errors'])}")
    print(args.report)
    for error in report['errors']: print(error)
    raise SystemExit(bool(report['errors']))

if __name__ == '__main__': main()
