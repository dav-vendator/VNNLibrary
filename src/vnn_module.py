#A simple class that represents NN
from typing import Any, Dict
from abc import ABC, abstractmethod

class VNNModule(ABC):
    """Abstract base class for neural network modules."""
    
    def __init__(self, name: str):
        import numpy as np
        super().__init__()
        self.module_name = name  # For debugging purposes
        self._output_size = np.NaN # Initialize output size as NaN

    @property
    @abstractmethod
    def output_size(self) -> Any:
        """Property to get the size of the output from this module."""
        return self._output_size
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}--{self.module_name}'
    
    def __call__(self, x: Any) -> Any:
        """Make the module callable."""
        return self.forward(x)
    
    @abstractmethod
    def summary(self) -> Dict[str, int]:
        """Summarize the parameters and structure of the module."""
        pass

    @abstractmethod
    def sanity_check(self, from_behind: Any) -> bool:
        """Checks whether input from the previous layer matches the expected shape."""
        pass

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Forward pass to process input x."""
        pass

    @abstractmethod
    def backward(self, from_front: Any) -> Any:
        """Backward pass for gradient calculation."""
        pass
