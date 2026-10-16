"""Explicit candidate snapshot component using the current run saving service."""

from pep_compass.optimization.engine.execution.step import Step


class Save(Step):
    """Capture a named batch without changing its computational meaning.

    :param name: Snapshot label within the current run.
    """

    def __init__(self, name="snapshot"):
        self.snapshot_name = name

    def _execute(self, batch, context):
        if context.saver is None:
            raise ValueError("A save component requires the Runner saving service.")
        context.saver.capture(batch, name=self.snapshot_name, context=context)
        return batch
