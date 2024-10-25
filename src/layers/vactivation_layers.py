import numpy as np
from typing import Dict
from ..vnn_module import VNNModule

class VSigmoidLayer(VNNModule):
    def __init__(self, output_size):
        super().__init__(f'VSigmoid Layer({output_size})')
        self.forward_acc = None
        self.backward_acc = None
        self._output_size = output_size

    @property
    def output_size(self):
          return self._output_size

    def forward(self, x: np.ndarray) -> np.ndarray:
            #forward pass 
            out = 1.0/(1.0+np.e**(-x))
            self.forward_acc = out
            return  out
    
    def clear_accumulations(self) -> None:
            self.forward_acc = None
            self.backward_acc = None
    
    def summary(self) -> Dict[str, int]:
          return {} #activation function

    def sanity_check(self, from_behind: VNNModule) -> bool:
          print(f'From Behind: {from_behind} and Forward: {self._output_size}')
          return from_behind.output_size == self._output_size #no need for checking in activation function
    
    def backward(self, from_front:np.ndarray) -> np.ndarray:
        sigmoid_grad = self.forward_acc * (1 - self.forward_acc)
        self.backward_acc = sigmoid_grad * from_front
        return self.backward_acc
    

#ReLu Activation Layer
class VReLuLayer(VNNModule):
    def __init__(self, output_size):
        # Initialize the layer
        super().__init__(f'VReLU Layer({output_size})')
        self.err_term = 0.00004  # Values smaller than this are treated as 0
        self.forward_acc = None  # Stores forward pass result
        self.backward_acc = None  # Stores backward pass result
        self._output_size = output_size
        
    @property
    def output_size(self):
          return self._output_size
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for ReLU."""
        out = np.maximum(0, x)  # NumPy's vectorized ReLU
        self.forward_acc = out  # Store for backward pass
        return out

    def clear_accumulations(self) -> None:
        """Clear stored forward and backward results."""
        self.forward_acc = None
        self.backward_acc = None

    def sanity_check(self, from_behind: VNNModule) -> bool:
        print(f'From Behind: {from_behind} and Forward: {self._output_size}')
        return from_behind.output_size == self._output_size #no need for checking in activation function
    
    def summary(self) -> Dict[str, int]:
        return {}
    
    def backward(self, from_front: np.ndarray) -> np.ndarray:
        """Backward pass for ReLU."""
        # Derivative of ReLU: 1 where forward_acc > err_term, else 0
        relu_grad = (self.forward_acc > self.err_term).astype(np.float32)
        
        # Element-wise multiplication with incoming gradient
        self.backward_acc = relu_grad * from_front