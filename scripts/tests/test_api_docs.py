"""Regression tests for shared API rendering and output completeness checks."""

import ast
import html
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest

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
'''


class SharedRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        root = Path(cls.directory.name)
        (root / 'docs').mkdir()
        (root / 'fixture.py').write_text(textwrap.dedent(FIXTURE))
        (root / 'docs/index.md').write_text('# Fixture\n\n::: fixture.Example\n')
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

    def test_property_keeps_native_sections_and_class_summary(self):
        for text in ['Class-level property summary.', 'Compute the current count.', 'Native property return details.', 'Native property exception details.', 'instance.count']:
            with self.subTest(text=text): self.assertIn(text, self.text)
        self.assertEqual(self.html.count('id="fixture.Example.count"'), 1)
        self.assertNotIn('data-field="count"', self.html)

    def test_detailed_field_keeps_own_documentation_and_class_summary(self):
        for text in ['Class-level detailed field summary.', 'Independent field documentation.', 'instance.detailed']:
            with self.subTest(text=text): self.assertIn(text, self.text)
        self.assertEqual(self.html.count('id="fixture.Example.detailed"'), 1)
        self.assertNotIn('data-field="detailed"', self.html)

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


if __name__ == '__main__':
    unittest.main()
