"""Configuration-driven PepCompass composition root."""

from pep_compass.core.builder import PepCompassCore
from pep_compass.core.configuration import load_configuration
from pep_compass.core.validation import validate_configuration

__all__ = ["PepCompassCore", "load_configuration", "validate_configuration"]
