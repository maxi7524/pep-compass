from abc import ABC, abstractmethod

class AbstractOptimizer(ABC):
    def __init__(self, black_box):
        self.black_box = black_box

    @abstractmethod
    def optimize(self, evaluation_budget: int, starting_point):
        """
        Optimize the black_box function.

        Args:
            evaluation_budget (int): Maximum number of evaluations allowed.
            starting_point (Any): Initial point for optimization.

        Returns:
            Any: The result of the optimization.
        """
        pass
