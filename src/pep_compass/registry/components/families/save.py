"""Typed registry for explicit snapshot components."""

from pep_compass.optimization.components.save.snapshot import Save
from pep_compass.registry.components.catalog import component_catalog
from pep_compass.registry.factories.registry import Registry

save_registry = Registry("save", expected_type=Save)
component_catalog.register_family("save", save_registry)
