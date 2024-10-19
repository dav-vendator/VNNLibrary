#A simple class that represents NN
from typing import Any, Dict
from abc import ABC, abstractmethod

class VNNModule(ABC):
    """An abstract class for all VNN modules """
    def __init__(self, name):
        super().__init__()
        self.module_name = name #for debugging purpose

    def __repr__(self) -> str:
        return  f'{self.__class__.__name__}--{self.module_name}' 
    
    def __call__(self, x: Any) -> Any:
        return self.forward(x)
    
    @abstractmethod
    def summary(self) -> Dict[str, int]:
        pass
    
    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Forward pass to process input x."""
        pass

    @abstractmethod
    def backward(self, from_front: Any) -> Any:
        """Backward pass for gradient calculation."""
        pass