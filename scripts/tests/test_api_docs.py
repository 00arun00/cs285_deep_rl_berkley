"""Regression tests for shared API rendering and output completeness checks."""

import ast
import contextlib
import io
import json
import html
from html.parser import HTMLParser
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
import zlib
from unittest.mock import patch

import griffe

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('check_docs', ROOT / 'scripts/check_docs.py')
check_docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_docs)

FIXTURE = '''
class Example:
    """Exercise shared field presentation.

    Attributes:
        plain: Plain compact summary.
        count: Class-level property summary.
        detailed: Class-level detailed field summary.
    """

    plain: int = 1
    detailed: int = 2
    """Independent field documentation.

    Examples:
        ```python
        instance.detailed = 3
        ```
    """

    @property
    def count(self):
        """Compute the current count.

        ## Notes

        Getter heading details.

        Returns:
            count (int): Native property return details.

        Raises:
            ValueError: Native property exception details.

        Examples:
            ```python
            print(instance.count)
            ```
        """
        return 3

    @count.setter
    def count(self, new_value: int):
        """Replace the stored count.

        ## Notes

        Setter heading details.

        Args:
            new_value: Setter replacement parameter details.

        Returns:
            result (int): Setter result details.

        Raises:
            ValueError: Setter exception details.

        Examples:
            ```python
            instance.count = 7
            ```
        """
        return new_value

    @count.deleter
    def count(self):
        """Clear the stored count.

        ## Notes

        Deleter heading details.

        Returns:
            result (int): Deleter result details.

        Raises:
            RuntimeError: Deleter exception details.

        Examples:
            ```python
            del instance.count
            ```
        """
        return 0

    @property
    def readonly(self):
        """Read-only property details.

        Returns:
            result (int): Read-only result details.
        """
        return 4
'''


FIELD_FIXTURE = '''
class Storage:
    """Store runtime values.

    Attributes:
        saved: Constructor-assigned value.
        left: Left tuple-assigned field.
        right: Right tuple-assigned field.
    """

    def __init__(self, value=4):
        self.saved = value
        self.left, self.right = (2, 3)

from dataclasses import dataclass

@dataclass
class Defaults:
    """Represent declared dataclass fields.

    Attributes:
        required: Field with no declared default.
        threshold: Field with a declared default.
    """

    required: int
    threshold: int = 5
'''


ENUM_FIXTURE = '''
from enum import Enum, IntEnum as IntegerEnum

class Registry(IntegerEnum):
    """Permanent responsibility IDs.

    Attributes:
        FIRST: First responsibility.
    """
    FIRST = 7

class Status(Enum):
    """Named status values.

    Attributes:
        READY: Ready status.
    """
    READY = "ready"
'''


class SharedRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        root = Path(cls.directory.name)
        (root / 'docs').mkdir()
        (root / 'fixture.py').write_text(textwrap.dedent(FIXTURE)+textwrap.dedent(FIELD_FIXTURE)+textwrap.dedent(ENUM_FIXTURE))
        (root / 'docs/index.md').write_text('# Fixture\n\n::: fixture.Example\n\n::: fixture.Storage\n\n::: fixture.Defaults\n\n::: fixture.Registry\n\n::: fixture.Status\n')
        (root / 'zensical.toml').write_text(f'''[project]
site_name = "API regression fixture"
[project.markdown_extensions]
pymdownx.superfences = {{}}
[project.plugins.mkdocstrings]
custom_templates = "{ROOT / 'docs-overrides'}"
[project.plugins.mkdocstrings.handlers.python]
paths = ["."]
[project.plugins.mkdocstrings.handlers.python.options]
allow_inspection = false
show_root_heading = true
show_source = false
show_if_no_docstring = true
show_docstring_attributes = false
merge_init_into_class = false
filters = []
docstring_section_style = "list"
[project.plugins.mkdocstrings.handlers.python.options.extra]
documented_fields = true
''')
        result = subprocess.run([sys.executable, '-m', 'zensical', 'build', '--strict'], cwd=root, text=True, capture_output=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.html = (root / 'site/index.html').read_text()
        cls.text = html.unescape(re.sub('<[^>]+>', '', cls.html))
        cls.inventory = zlib.decompress((root/'site/objects.inv').read_bytes().split(b'\n', 4)[4]).decode()

    def test_property_keeps_native_sections_and_class_summary(self):
        for text in ['Class-level property summary.', 'Compute the current count.', 'Native property return details.', 'Native property exception details.', 'instance.count']:
            with self.subTest(text=text): self.assertIn(text, self.text)
        self.assertEqual(self.html.count('id="fixture.Example.count"'), 1)
        self.assertNotIn('data-field="count"', self.html)

    def accessor_text(self, kind):
        match = re.search(r'<section class="doc-property-accessor" data-accessor="'+kind+r'" data-property="fixture.Example.count"[^>]*>(.*?)</section>', self.html, re.S)
        self.assertIsNotNone(match, 'Missing '+kind+' section')
        return html.unescape(re.sub('<[^>]+>', '', match.group(1)))

    def test_setter_sections_render_inside_setter_container(self):
        text = self.accessor_text('setter')
        for marker in ['Setter', 'Replace the stored count.', 'new_value', 'Setter replacement parameter details.', 'Setter result details.', 'Setter exception details.', 'instance.count = 7']:
            with self.subTest(marker=marker): self.assertIn(marker, text)
        self.assertNotIn('Clear the stored count.', text)

    def test_deleter_sections_render_inside_deleter_container(self):
        text = self.accessor_text('deleter')
        for marker in ['Deleter', 'Clear the stored count.', 'Deleter result details.', 'Deleter exception details.', 'del instance.count']:
            with self.subTest(marker=marker): self.assertIn(marker, text)
        self.assertNotIn('Replace the stored count.', text)

    def test_accessors_keep_one_canonical_property_inventory_entry(self):
        entries = [line for line in self.inventory.splitlines() if line.split()[0] == 'fixture.Example.count']
        self.assertEqual(len(entries), 1, entries)
        self.assertEqual(self.html.count('id="fixture.Example.count"'), 1)
        self.assertNotIn('fixture.Example.count.setter ', self.inventory)
        self.assertNotIn('fixture.Example.count.deleter ', self.inventory)

    def test_accessor_markdown_headings_have_distinct_namespaces(self):
        ids = re.findall(r'id="(fixture\.Example\.count[^"]*notes[^"]*)"', self.html)
        self.assertEqual(len(ids), 3, ids)
        self.assertEqual(len(set(ids)), 3, ids)
        self.assertTrue(any('--setter' in value for value in ids), ids)
        self.assertTrue(any('--deleter' in value for value in ids), ids)

    def test_accessor_toc_has_distinct_parents_for_inner_headings(self):
        class TocParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.parents, self.links = [], []

            def handle_starttag(self, tag, attributes):
                attrs = dict(attributes)
                if tag == 'nav':
                    self.parents.append(attrs.get('aria-label'))
                elif tag == 'a' and 'md-nav__link' in attrs.get('class', '').split():
                    self.links.append((attrs.get('href'), self.parents[-1] if self.parents else None))

            def handle_endtag(self, tag):
                if tag == 'nav': self.parents.pop()

        toc = TocParser()
        toc.feed(self.html)
        expected = {
            '#fixture.Example.count--setter': 'count',
            '#fixture.Example.count--deleter': 'count',
            '#fixture.Example.count--setter--notes': 'Setter',
            '#fixture.Example.count--deleter--notes': 'Deleter',
        }
        for href, parent in expected.items():
            with self.subTest(href=href):
                self.assertEqual({label for target, label in toc.links if target == href}, {parent})
        targets = [target for target, _ in toc.links]
        self.assertLess(targets.index('#fixture.Example.count--setter'), targets.index('#fixture.Example.count--setter--notes'))
        self.assertLess(targets.index('#fixture.Example.count--setter--notes'), targets.index('#fixture.Example.count--deleter'))
        self.assertNotIn('fixture.Example.count--setter ', self.inventory)
        self.assertNotIn('fixture.Example.count--deleter ', self.inventory)

    def test_readonly_property_has_no_accessor_sections(self):
        self.assertIn('Read-only property details.', self.text)
        self.assertIn('Read-only result details.', self.text)
        self.assertEqual(self.html.count('id="fixture.Example.readonly"'), 1)
        self.assertNotIn('data-property="fixture.Example.readonly"', self.html)

    def test_detailed_field_keeps_own_documentation_and_class_summary(self):
        for text in ['Class-level detailed field summary.', 'Independent field documentation.', 'instance.detailed']:
            with self.subTest(text=text): self.assertIn(text, self.text)
        self.assertEqual(self.html.count('id="fixture.Example.detailed"'), 1)
        self.assertNotIn('data-field="detailed"', self.html)

    def field_row(self, name):
        match = re.search(r'<section class="documented-field" data-field="'+name+r'"[^>]*>(.*?)</section>', self.html, re.S)
        self.assertIsNotNone(match, name)
        return match.group(1)

    def test_unextracted_tuple_fields_keep_anchors_without_invented_metadata(self):
        for name in ['left', 'right']:
            row = self.field_row(name)
            self.assertIn('tuple-assigned field.', row)
            self.assertNotIn('documented-field__metadata', row)
            self.assertEqual(self.html.count('id="fixture.Storage.'+name+'"'), 1)

    def test_constructor_value_is_not_presented_as_default(self):
        row = self.field_row('saved')
        self.assertIn('<dt>Assigned value</dt>', row)
        self.assertIn('<code>value</code>', row)
        self.assertNotIn('<dt>Default</dt>', row)
        self.assertNotIn('<dt>Type</dt>', row)

    def test_dataclass_default_and_required_metadata_are_preserved(self):
        row = self.field_row('threshold')
        self.assertIn('<dt>Default</dt>', row)
        self.assertIn('<code>5</code>', row)
        self.assertNotIn('Assigned value', row)
        row = self.field_row('required')
        self.assertIn('<dt>Default</dt>', row)
        self.assertIn('No default', row)

    def test_enum_members_render_values_not_constructor_defaults(self):
        for class_name, member, value in [('Registry', 'FIRST', '7'), ('Status', 'READY', 'ready')]:
            with self.subTest(member=member):
                row = self.field_row(member)
                self.assertIn('<dt>Value</dt>', row)
                self.assertIn(value, row)
                self.assertNotIn('<dt>Default</dt>', row)
                self.assertNotIn('Assigned value', row)
                self.assertEqual(self.html.count('id="fixture.'+class_name+'.'+member+'"'), 1)
        self.assertIn('<dt>Default</dt>', self.field_row('plain'))
        self.assertNotIn('<dt>Value</dt>', self.field_row('plain'))

    def test_plain_field_retains_compact_presentation(self):
        self.assertIn('data-field="plain"', self.html)
        self.assertIn('Plain compact summary.', self.html)
        self.assertEqual(self.html.count('id="fixture.Example.plain"'), 1)


class OutputCompletenessTests(unittest.TestCase):
    def check_method(self, body, doc='Summary.', annotation='', decorator=''):
        source = 'class Example:\n' + (f'    @{decorator}\n' if decorator else '')
        source += f'    def value(self){annotation}:\n'
        source += textwrap.indent('"""' + doc + '\n"""\n' + body, '        ')
        node = ast.parse(source).body[0].body[0]
        module = griffe.visit('fixture', filepath=Path('fixture.py'), code=source, docstring_parser=griffe.Parser.google)
        return check_docs.output_doc_issues(node, module['Example.value'])

    def test_setter_does_not_inherit_getter_return_requirement(self):
        source = '''class Example:
    @property
    def value(self):
        """Getter summary."""
        return 3
    @value.setter
    def value(self, new_value):
        """Setter summary."""
        self._value = new_value
'''
        node = ast.parse(source).body[0].body[-1]
        module = griffe.visit('fixture', filepath=Path('fixture.py'), code=source, docstring_parser=griffe.Parser.google)
        issues, _ = check_docs.output_doc_issues(node, module['Example.value'])
        self.assertNotIn('Missing return description: value', issues)

    def test_missing_docstring_still_requires_explicit_output_docs(self):
        source = 'class Example:\n    def value(self):\n        return 3\n'
        node = ast.parse(source).body[0].body[0]
        module = griffe.visit('fixture', filepath=Path('fixture.py'), code=source, docstring_parser=griffe.Parser.google)
        issues, _ = check_docs.output_doc_issues(node, module['Example.value'])
        self.assertIn('Missing return description: value', issues)

    def test_async_own_value_return_requires_description(self):
        source = 'class Example:\n    async def value(self):\n        return await self.fetch()\n'
        node = ast.parse(source).body[0].body[0]
        module = griffe.visit('fixture', filepath=Path('fixture.py'), code=source, docstring_parser=griffe.Parser.google)
        issues, _ = check_docs.output_doc_issues(node, module['Example.value'])
        self.assertIn('Missing return description: value', issues)

    def test_unannotated_value_return_requires_description(self):
        issues, _ = self.check_method('return 1')
        self.assertIn('Missing return description: value', issues)

    def test_unannotated_property_requires_description(self):
        issues, _ = self.check_method('raise NotImplementedError', decorator='property')
        self.assertIn('Missing return description: value', issues)

    def test_qualified_property_requires_description(self):
        issues, _ = self.check_method('raise NotImplementedError', decorator='builtins.property')
        self.assertIn('Missing return description: value', issues)

    def test_bare_none_and_nested_returns_are_not_outputs(self):
        for body in ['return', 'return None', 'pass', 'def nested():\n    return 3\nreturn None', 'class Nested:\n    def value(self):\n        return 3\nreturn None', 'callback = lambda: 3']:
            with self.subTest(body=body):
                issues, _ = self.check_method(body)
                self.assertNotIn('Missing return description: value', issues)

    def test_own_conditional_return_is_detected(self):
        issues, _ = self.check_method('if self.ready:\n    return self.value')
        self.assertIn('Missing return description: value', issues)

    def test_none_annotation_does_not_hide_explicit_value(self):
        issues, _ = self.check_method('return 3', annotation=' -> None')
        self.assertIn('Missing return description: value', issues)

    def test_documented_unannotated_return_passes(self):
        issues, _ = self.check_method('return 3', doc='Summary.\n\nReturns:\n    result (int): Current value.')
        self.assertEqual(issues, [])

    def test_generator_requires_yields_not_annotation_returns(self):
        issues, _ = self.check_method('yield 1', annotation=' -> Iterator[int]')
        self.assertIn('Missing yield description: value', issues)
        self.assertNotIn('Missing return description: value', issues)

    def test_yield_from_and_terminal_return_are_separate(self):
        issues, ambiguities = self.check_method('yield from self.items\nreturn 4')
        self.assertIn('Missing yield description: value', issues)
        self.assertIn('Missing return description: value', issues)
        self.assertTrue(ambiguities)

    def test_nested_yield_does_not_make_outer_generator(self):
        issues, _ = self.check_method('def nested():\n    yield 1\nreturn 2')
        self.assertIn('Missing return description: value', issues)
        self.assertNotIn('Missing yield description: value', issues)

    def test_return_expression_uncertainty_is_reported(self):
        issues, ambiguities = self.check_method('return self.maybe_none()')
        self.assertIn('Missing return description: value', issues)
        self.assertTrue(ambiguities)

    def test_documented_generator_passes(self):
        issues, _ = self.check_method('yield 1', doc='Summary.\n\nYields:\n    item (int): Next item.')
        self.assertEqual(issues, [])


class AccessorIntegrationTests(unittest.TestCase):
    def run_checker(self, setter_doc=None, setter_body='pass', deleter_doc='Remove the value.', deleter_body='pass', getter_doc=None, extra_attribute=None):
        if setter_doc is None:
            setter_doc = 'Set the value.\n\nArgs:\n    new_value (int): Replacement value.'
        if getter_doc is None:
            getter_doc = 'Read the value.\n\nReturns:\n    result (int): Current value.'
        source = 'class Example:\n    """A separately documented property.\n\n    Attributes:\n        value: Shared property summary.\n    """\n'
        if extra_attribute:
            source = source.replace('        value: Shared property summary.', '        value: Shared property summary.\n        '+extra_attribute+': Stale field.')
        for decorator, signature, doc, body in [
            ('property', 'value(self)', getter_doc, 'return 1'),
            ('value.setter', 'value(self, new_value)', setter_doc, setter_body),
            ('value.deleter', 'value(self)', deleter_doc, deleter_body),
        ]:
            source += f'\n    @{decorator}\n    def {signature}:\n'
            if doc:
                source += textwrap.indent('"""'+doc+'\n"""\n', '        ')
            source += textwrap.indent(body+'\n', '        ')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / 'hw1/src/hw1_imitation'
            package.mkdir(parents=True)
            (package/'__init__.py').write_text('')
            (package/'data.py').write_text(source)
            (root/'docs').mkdir()
            (root/'docs/index.md').write_text('# Fixture\n')
            (root/'zensical.toml').write_text(f'''[project]
site_name = "Accessor integration"
[project.plugins.mkdocstrings]
custom_templates = "{ROOT / 'docs-overrides'}"
[project.plugins.mkdocstrings.handlers.python]
paths = ["hw1/src"]
[project.plugins.mkdocstrings.handlers.python.options]
allow_inspection = false
show_root_heading = true
show_source = false
show_if_no_docstring = true
show_docstring_attributes = false
filters = []
[project.plugins.mkdocstrings.handlers.python.options.extra]
documented_fields = true
[project.plugins.api-autonav]
modules = ["hw1/src/hw1_imitation"]
api_root_uri = "api"
''')
            build = subprocess.run([sys.executable, '-m', 'zensical', 'build', '--strict'], cwd=root, text=True, capture_output=True)
            self.assertEqual(build.returncode, 0, build.stdout+build.stderr)
            page = (root/'site/api/hw1_imitation/data/index.html').read_text()
            self.assertEqual(page.count('id="hw1_imitation.data.Example.value"'), 1)
            report_path = root/'report.json'
            argv = ['check_docs.py', '--pilot', 'hw1_imitation.data.Example', '--ruff', 'ruff', '--report', str(report_path)]
            # Isolate the external style tool only. Run real discovery, Griffe
            # extraction, native HTML reconciliation and all completeness logic.
            with patch.object(check_docs, 'ROOT', root), patch.object(check_docs, 'SOURCE', root/'hw1/src'), patch.object(sys, 'argv', argv), patch.object(check_docs.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '', '')), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as outcome:
                    check_docs.main()
            return outcome.exception.code, json.loads(report_path.read_text())

    def test_correct_accessors_pass_full_checker(self):
        code, report = self.run_checker()
        self.assertFalse(code, report['errors'])
        self.assertEqual(report['classes'][0]['members'], ['value.getter', 'value.setter', 'value.deleter'])

    def test_getter_return_docs_are_not_borrowed_from_setter(self):
        code, report = self.run_checker(
            getter_doc='Read the value.',
            setter_doc='Set the value.\n\nArgs:\n    new_value (int): Replacement value.\n\nReturns:\n    result (int): Setter result.',
            setter_body='return 3',
        )
        self.assertTrue(code)
        self.assertIn('Missing return description: value.getter', report['errors'])
        self.assertNotIn('Missing return description: value.setter', report['errors'])

    def test_setter_parameters_use_setter_docs(self):
        code, report = self.run_checker(setter_doc='Set the value.\n\nArgs:\n    obsolete (int): Stale parameter.')
        self.assertTrue(code)
        self.assertIn('Missing parameter: value.setter.new_value', report['errors'])
        self.assertIn('Stale parameter: value.setter.obsolete', report['errors'])

    def test_setter_missing_docs_are_not_borrowed(self):
        code, report = self.run_checker(setter_doc='')
        self.assertTrue(code)
        self.assertIn('Missing member docstring: value.setter', report['errors'])
        self.assertIn('Missing parameter: value.setter.new_value', report['errors'])

    def test_setter_return_docs_are_not_borrowed_from_getter(self):
        code, report = self.run_checker(setter_body='return 3')
        self.assertTrue(code)
        self.assertIn('Missing return description: value.setter', report['errors'])

    def test_deleter_return_docs_are_not_borrowed_from_getter(self):
        code, report = self.run_checker(deleter_body='return 3')
        self.assertTrue(code)
        self.assertIn('Missing return description: value.deleter', report['errors'])

    def test_deleter_missing_docs_are_identified(self):
        code, report = self.run_checker(deleter_doc='')
        self.assertTrue(code)
        self.assertIn('Missing member docstring: value.deleter', report['errors'])

    def test_fallback_does_not_hide_stale_attribute_from_checker(self):
        code, report = self.run_checker(extra_attribute='ghost')
        self.assertTrue(code)
        self.assertIn('Stale attribute: ghost', report['errors'])

    def test_unresolved_accessor_does_not_fall_back_to_getter(self):
        source = 'class Example:\n    @property\n    def value(self):\n        return 1\n    @value.setter\n    def value(self, new_value):\n        pass\n'
        node = ast.parse(source).body[0].body[-1]
        module = griffe.visit('fixture', filepath=Path('fixture.py'), code=source, docstring_parser=griffe.Parser.google)
        module['Example.value'].setter = None
        member, label, issue = check_docs.resolve_own_member(module['Example'], node)
        self.assertIsNone(member)
        self.assertEqual(label, 'value.setter')
        self.assertIn('Cannot resolve own member documentation', issue)

if __name__ == '__main__':
    unittest.main()
