import numpy as np
from typing import Tuple
from vnn_module import VNNModule

class VLinearLayer(VNNModule):
    def __init__(self, input_size:int, output_size: int, weight_initialize: np.ndarray = None, bias_initialize: np.ndarray = None):
        super().__init__(f'VLinearLayer mapping {input_size} -> {output_size}')
        self.input_size = input_size
        self.output_size = output_size
        #TODO: Replace random initialization with standard techniques
        self.WeightMatrix = np.random.rand(self.input_size, self.output_size) if weight_initialize is None else weight_initialize
        self.bias = np.random.rand(self.output_size) if bias_initialize is None else bias_initialize 
        #sanity check 
        if self.WeightMatrix.shape != (input_size, output_size):
            raise ValueError('Input and output sizes did not matched with the weight matrix size.')
        if self.bias.shape != (output_size,):
            raise ValueError('Output size did not matched with the bias vector size.')
        #for computation of gradients
        self.forward_acc = None
        self.backward_acc = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        #forward pass is simple to implement
        #Linear layer has equation == f(x) = W.x + b
        if x.shape[0] != self.input_size:
            raise ValueError(f"Input size {x.shape[0]} does not match expected input size {self.input_size}.")
        out = np.dot(x, self.WeightMatrix)
        out += self.bias
        self.forward_acc  = out
        return out

    def clear_accumulations(self) -> None:
        self.forward_acc = None
        self.backward_acc = None

    def backward(self, from_front: np.ndarray) -> Tuple[np.ndarray]:
        #compute gradient of weights and biases and multiply them with from front
        pass