import numpy as np
from vnn_module import VNNModule

class VSigmoidLayer(VNNModule):
    def __init__(self):
        super().__init__(f'VSigmoid Layer')
        self.sigmoid = np.vectorize(lambda x: 1.0/(1+np.exp(-x)))
        self.forward_acc = None
        self.backward_acc = None

    def forward(self, x: np.ndarray) -> np.ndarray:
            #forward pass 
            out = self.sigmoid(x)
            self.forward_acc = out
            return  out
    
    def clear_accumulations(self) -> None:
            self.forward_acc = None
            self.backward_acc = None
    
    def backward(self, from_front:np.ndarray) -> np.ndarray:
        sigmoid_grad = self.forward_acc * (1 - self.forward_acc)
        self.backward_acc = sigmoid_grad * from_front
        return self.backward_acc