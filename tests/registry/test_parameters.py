"""Canonical registry defaults and strict typed parameter validation."""

from dataclasses import dataclass

import pytest

from pep_compass.registry.factories.registry import Registry


def test_typed_registration_defaults_and_validation_are_shared():
    @dataclass
    class Configuration:
        value: int = 3

    registry = Registry("typed")

    @registry.register("method", config_type=Configuration)
    def build(value=0):
        return value

    assert registry.build("method") == 3
    assert registry.describe("method")["parameters"][0]["default"] == 3
    with pytest.raises(ValueError, match="Unknown"):
        registry.build("method", {"unknown": 1})
    with pytest.raises(ValueError, match="must match"):
        registry.build("method", {"value": True})


def test_registered_defaults_are_immutable_and_copied():
    registry = Registry("defaults")
    original = {"values": {"count": 3}}

    @registry.register("method", defaults=original)
    def build(values):
        return values

    original["values"]["count"] = 100
    assert registry.build("method") == {"count": 3}
    with pytest.raises(TypeError):
        registry.definition("method").defaults["values"]["count"] = 4


def test_registered_defaults_override_typed_defaults_in_description_and_build():
    @dataclass
    class Configuration:
        value: int = 3

    registry = Registry("typed override")

    @registry.register("method", config_type=Configuration, defaults={"value": 7})
    def build(value):
        return value

    assert registry.build("method") == 7
    assert registry.describe("method")["parameters"][0]["default"] == 7
