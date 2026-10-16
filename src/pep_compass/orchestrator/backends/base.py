"""Contract for dispatch strategies using the public wrapper worker."""

from abc import ABC, abstractmethod


class ExecutionBackend(ABC):
    """Dispatch independent jobs; computational execution belongs to Pipeline."""

    @abstractmethod
    def run(self, orchestrator, plan, plan_path):
        """Return terminal job records or a generated scheduler-script path."""

    @classmethod
    def resolve_parameters(cls, parameters, working_directory):
        """Resolve backend-owned resources before freezing a plan.

        :param parameters: Validated factory parameters.
        :param working_directory: Base directory for configuration-relative paths.
        :return: Portable parameter mapping.
        """
        return parameters
