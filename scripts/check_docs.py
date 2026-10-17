#!/usr/bin/env python3
"""Reconcile native API output with source and enforce one documentation pilot."""
import argparse
import ast
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile

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
                    yield from walk_classes([v for v in value if isinstance(v, ast.stmt)], prefix, local)


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


def parameters(node):
    args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    args += [a for a in (node.args.vararg, node.args.kwarg) if a]
    return {a.arg for a in args if a.arg not in {'self', 'cls'}}


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', default=PILOT, help='Qualified class to enforce; other classes remain report-only')
    parser.add_argument('--report', type=Path, default=Path('/tmp/normalizer-pilot/coverage.json'))
    parser.add_argument('--baseline', type=Path, help='Optional pre-edit pilot module for executable AST comparison')
    parser.add_argument('--ruff', default=os.environ.get('RUFF_BIN', shutil.which('ruff')))
    args = parser.parse_args()
    package = griffe.load('hw1_imitation', search_paths=[str(SOURCE)], allow_inspection=False, docstring_parser='google')
    report = {'classes': [], 'exceptions': [], 'errors': [], 'limits': [
        'Static assignments cannot establish arbitrary runtime-created attributes.',
        'Semantic accuracy requires source and rendered review; structural checks do not prove it.',
        'Inherited and imported members are excluded from authored coverage.',
    ]}
    native = set()
    native_classes = []
    pilot_node = None
    for page in (ROOT/'site/api').rglob('index.html'):
        # Native Zensical module discovery creates each module page. Verify actual output,
        # not a second hand-maintained class/page inventory.
        html = page.read_text()
        native.update(re.findall(r'id="(hw1_imitation\.[^"]+)"', html))
        native_classes.extend(re.findall(r'<div class="doc doc-object doc-class">.*?<h[1-6][^>]*id="([^"]+)"', html, re.S))
    for file in sorted(SOURCE.rglob('*.py')):
        rel = file.relative_to(SOURCE).with_suffix('')
        parts = list(rel.parts)
        if parts[-1] == '__init__': parts.pop()
        module = '.'.join(parts)
        tree = ast.parse(file.read_text())
        for path, node, local in walk_classes(tree.body, module):
            if local:
                report['exceptions'].append({'path': path, 'reason': 'function-local class'})
                continue
            obj = package[path.removeprefix('hw1_imitation.')]
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            fields = set()
            for n in own_nodes(node.body):
                if isinstance(n, (ast.Assign, ast.AnnAssign)):
                    for target in n.targets if isinstance(n, ast.Assign) else [n.target]:
                        fields.update(assigned_names(target))
            for method in methods:
                for n in own_nodes(method.body):
                    if isinstance(n, (ast.Assign, ast.AnnAssign)):
                        targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                        fields.update(set().union(*(assigned_names(t, True) for t in targets)))
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {'setattr', 'exec'}:
                        report['exceptions'].append({'path': path+'.'+method.name, 'reason': 'dynamic assignment requires review'})
            issues=[]
            if not (ast.get_docstring(node) or "").strip(): issues.append('Missing class docstring')
            attrs = documented(obj, 'attributes')
            for name in fields:
                if name in obj.members and obj[name].docstring and obj[name].docstring.value.strip():
                    attrs.setdefault(name, obj[name].docstring.value)
            for name in sorted(fields - attrs.keys()): issues.append('Missing attribute description: '+name)
            property_names={n.name for n in methods if any(isinstance(d, ast.Name) and d.id=='property' for d in n.decorator_list)}
            for name in sorted(attrs.keys()-fields-property_names): issues.append('Stale attribute: '+name)
            for name, desc in attrs.items():
                if not desc.strip(): issues.append('Empty attribute description: '+name)
            member_labels = []
            for method in methods:
                member, label, resolution_issue = resolve_own_member(obj, method)
                member_labels.append(label)
                if not (ast.get_docstring(method) or "").strip(): issues.append('Missing member docstring: '+label)
                if resolution_issue:
                    issues.append(resolution_issue)
                    report['exceptions'].append({'path': path+'.'+label, 'reason': 'Native accessor metadata could not be matched to its own source; manual review required'})
                    continue
                params=documented(member,'parameters')
                output_issues, output_ambiguities = output_doc_issues(method, member)
                issues.extend(issue.replace(': '+method.name, ': '+label) for issue in output_issues)
                report['exceptions'].extend({'path': path+'.'+label, 'reason': reason} for reason in output_ambiguities)
                for name in sorted(parameters(method)-params.keys()): issues.append('Missing parameter: '+label+'.'+name)
                for name in sorted(params.keys()-parameters(method)): issues.append('Stale parameter: '+label+'.'+name)
                for name,desc in params.items():
                    if not desc.strip(): issues.append('Empty parameter: '+label+'.'+name)
            expected={path, *[path+'.'+n.name for n in methods], *[path+'.'+name for name in fields]}
            missing=sorted(expected-native)
            # Native renderer may omit undocumented instance assignments elsewhere;
            # report those separately, enforce complete pilot rendering.
            row={'path':path,'file':str(file.relative_to(ROOT)), 'members':member_labels, 'attributes':sorted(fields), 'issues':issues,'missing_rendered_anchors':missing}
            report['classes'].append(row)
            if path not in native: report['errors'].append('Native discovery omitted class: '+path)
            if path == args.pilot:
                report['errors'] += issues + ['Missing rendered pilot anchor: '+x for x in missing]
                pilot_node=node
                pilot_file=file
    def collected_classes(obj):
        for child in obj.members.values():
            if child.is_alias: continue
            if child.is_class: yield child.path
            if child.is_class or child.is_module: yield from collected_classes(child)
    ast_classes={r['path'] for r in report['classes']}
    griffe_classes=set(collected_classes(package))
    report['native_class_difference'] = {'missing': sorted(ast_classes-set(native_classes)), 'extra': sorted(set(native_classes)-ast_classes), 'duplicates': sorted({x for x in native_classes if native_classes.count(x)>1})}
    if any(report['native_class_difference'].values()): report['errors'].append('Native rendered class inventory differs from source')
    report['extraction_difference']={'missing':sorted(ast_classes-griffe_classes),'extra':sorted(griffe_classes-ast_classes)}
    if ast_classes != griffe_classes: report['errors'].append('AST and native-handler extraction class inventories differ')
    if args.baseline and pilot_node is not None:
        def executable(path):
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str): node.body=node.body[1:]
            return ast.dump(tree,include_attributes=False)
        report['executable_ast_unchanged']=executable(args.baseline)==executable(pilot_file)
        if not report['executable_ast_unchanged']: report['errors'].append('Executable AST changed')
    if pilot_node is None: report['errors'].append('Pilot class not found in source inventory')
    if not args.ruff: report['errors'].append('Ruff not found: set RUFF_BIN or pass --ruff; no environment is installed automatically')
    elif pilot_node is not None:
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'pilot.py';p.write_text(ast.get_source_segment(pilot_file.read_text(), pilot_node)+'\n')
            result=subprocess.run([args.ruff,'check','--config',str(ROOT/'docs/ruff.toml'),str(p)],text=True,capture_output=True)
            report['ruff']={'passed':result.returncode==0,'output':result.stdout+result.stderr}
            if result.returncode: report['errors'].append('Pilot Ruff docstring checks failed')
    report['pilot'] = args.pilot
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(f"Discovered {len(report['classes'])} source classes; pilot: {args.pilot}; errors: {len(report['errors'])}")
    print(args.report)
    for error in report['errors']: print(error)
    raise SystemExit(bool(report['errors']))

if __name__=='__main__': main()
