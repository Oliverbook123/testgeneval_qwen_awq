"""Regression coverage for dependencies in legacy Sphinx's fresh tox envs."""

import configparser
from pathlib import Path
import tempfile
import unittest

from swebench_docker.context_manager import TaskEnvContextManager


TOX_CONFIG = """[tox]
envlist = py39

[testenv]
deps =
    pytest
    du12: docutils==0.12
extras = test
commands = python -X dev -m pytest --durations 25 {posargs}
"""


class NullLog:
    def write(self, message):
        pass


class SphinxToxTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="testgeneval-tox-test-")
        self.addCleanup(self.temporary.cleanup)
        self.tox_path = Path(self.temporary.name) / "tox.ini"
        self.tox_path.write_text(TOX_CONFIG)
        # The config edit does not require a checked-out task or a real Conda env.
        self.context = TaskEnvContextManager.__new__(TaskEnvContextManager)
        self.context.log = NullLog()
        self.context.instance = {"repo": "sphinx-doc/sphinx", "version": "3.5"}

    def read_config(self):
        config = configparser.ConfigParser()
        config.read(self.tox_path)
        return config

    def apply(self):
        self.context.add_coverage_tox(str(self.tox_path))
        return self.read_config()

    def test_legacy_versions_constrain_actual_tox_testenv(self):
        for version in ("3.2", "3.5"):
            with self.subTest(version=version):
                self.context.instance["version"] = version
                config = self.apply()
                self.assertIn("setuptools<82", config.get("testenv", "deps").split())
                self.assertEqual(config.get("testenv", "extras"), "test")

    def test_preserves_multiline_dependencies_and_coverage_commands(self):
        config = self.apply()
        deps = config.get("testenv", "deps")
        self.assertIn("pytest", deps)
        self.assertIn("du12: docutils==0.12", deps)
        self.assertIn("setuptools<82", deps)
        self.assertEqual(
            config.get("testenv", "commands").splitlines(),
            ["python -X dev -m pytest --cov sphinx --durations 25 {posargs}",
             "coverage json -o coverage.json"],
        )

    def test_repeated_edits_do_not_duplicate_dependency_constraint(self):
        self.apply()
        config = self.apply()
        self.assertEqual(config.get("testenv", "deps").count("setuptools<82"), 1)
        self.assertEqual(config.get("testenv", "commands").count("--cov sphinx"), 1)

    def test_existing_stricter_setuptools_pin_is_not_replaced(self):
        self.tox_path.write_text(TOX_CONFIG.replace("    pytest", "    pytest\n    setuptools==68.0.0"))
        config = self.apply()
        self.assertIn("setuptools==68.0.0", config.get("testenv", "deps"))
        self.assertEqual(config.get("testenv", "deps").count("setuptools<82"), 1)

    def test_existing_compatibility_requirement_is_not_duplicated(self):
        self.tox_path.write_text(TOX_CONFIG.replace("    pytest", "    pytest\n    setuptools < 82"))
        config = self.apply()
        normalized = config.get("testenv", "deps").replace(" ", "")
        self.assertEqual(normalized.count("setuptools<82"), 1)

    def test_child_override_has_constraint_and_inherited_deps_are_not_duplicated(self):
        self.tox_path.write_text(TOX_CONFIG + """
[testenv:py39]
deps = custom-dependency

[testenv:coverage]
deps =
    {[testenv]deps}
    pytest-cov
""")
        config = self.apply()
        self.assertIn("setuptools<82", config.get("testenv:py39", "deps"))
        self.assertIn("custom-dependency", config.get("testenv:py39", "deps"))
        self.assertNotIn("setuptools", config.get("testenv:coverage", "deps"))
        self.assertIn("{[testenv]deps}", config.get("testenv:coverage", "deps"))

    def test_non_sphinx_and_other_versions_keep_dependencies(self):
        for instance in (
            {"repo": "pytest-dev/pytest", "version": "3.5"},
            {"repo": "sphinx-doc/sphinx", "version": "7.1"},
        ):
            with self.subTest(instance=instance):
                self.tox_path.write_text(TOX_CONFIG)
                before = self.read_config().get("testenv", "deps")
                self.context.instance = instance
                config = self.apply()
                self.assertEqual(config.get("testenv", "deps"), before)


if __name__ == "__main__":
    unittest.main()
