"""Domain-specific exceptions used by the API and pipeline."""


class CampusForecastError(Exception):
    """Base class for expected application errors."""


class DataValidationError(CampusForecastError):
    """Raised when a source table violates the canonical data contract."""


class DataUnavailableError(CampusForecastError):
    """Raised when a requested experiment lacks valid source coverage."""


class LeakageError(CampusForecastError):
    """Raised when a feature or split would expose unavailable information."""


class OptionalDependencyError(CampusForecastError):
    """Raised when an optional research dependency is needed but unavailable."""
