from abc import ABC, abstractmethod

class BlackBox(ABC):
    @abstractmethod
    def __call__(self, x):
        """
        Evaluate the black box function at input x.

        Args:
            x: Input parameters (could be a list, numpy array, etc.)

        Returns:
            The output of the black box function.
        """
        pass
    