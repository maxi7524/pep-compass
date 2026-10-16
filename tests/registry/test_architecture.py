"""Structural regression checks for the agreed ownership and import contracts."""

import ast
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "pep_compass"


def test_registry_instances_are_owned_only_by_registry_modules():
    misplaced = []
    for path in PACKAGE.rglob("*.py"):
        if path.is_relative_to(PACKAGE / "registry"):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"Registry", "ModelRegistry", "ComponentCatalog"}
            ):
                misplaced.append((str(path.relative_to(PACKAGE)), node.lineno))
    assert misplaced == []


def test_registry_does_not_import_core_and_has_no_empty_or_forwarding_modules():
    for path in (PACKAGE / "registry").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        meaningful = [
            node
            for node in tree.body
            if not (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            )
        ]
        assert meaningful, path
        assert not all(
            isinstance(node, (ast.Import, ast.ImportFrom)) for node in meaningful
        ), path
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("pep_compass.core"), path


def test_retired_namespaces_and_component_registry_files_are_absent():
    retired = [
        "data",
        "runtime",
        "registry/base.py",
        "registry/core.py",
        "registry/bootstrap.py",
        "registry/definitions.py",
        "registry/families",
        "autoencoder/registry.py",
        "predefined/registry.py",
        "orchestrator/backends/registry.py",
        "utils/strategy_factory.py",
        "core/specification.py",
        "core/validation.py",
        "optimization/engine/execution/result.py",
    ]
    for family in ("walkers", "mutation_generators", "filters", "oracles"):
        retired.append(f"optimization/components/{family}/manager.py")
    for relative in retired:
        assert not (PACKAGE / relative).exists(), relative
    assert list((PACKAGE / "optimization/components").rglob("*registry.py")) == []


def test_selected_builtin_loading_is_idempotent_and_does_not_load_oracles():
    script = """
from pep_compass.registry.components.loading import load_components
from pep_compass.registry.components.catalog import component_catalog
load_components("walker")
first = component_catalog.describe("walker")
load_components("walker")
assert component_catalog.describe("walker") == first
assert component_catalog.names() == ("walker",)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_builder_does_not_load_or_select_a_global_catalog():
    for path in (PACKAGE / "core/builder").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").endswith(
                    "registry.components.loading"
                ), path
                assert not any(
                    alias.name == "component_catalog" for alias in node.names
                ), path
