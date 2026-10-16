"""Reusable pipeline declarations and decorator-based preset discovery."""

from pep_compass.predefined.default_params import default_autoencoder
from pep_compass.predefined.factory import create
from pep_compass.registry.pipelines.catalog import list_predefined, register

__all__ = ["default_autoencoder", "create", "list_predefined", "register"]
