from ..vnn_module import VNNModule
from typing import List, Dict, Any

class VSequence(VNNModule):
    """
    VSequence orchestrates the execution of layers sequentially,
    handling both forward and backward passes.
    """

    def __init__(self, layers: List[VNNModule]):
        super().__init__(f'VSequence--{len(layers)}')
        self._output_size = layers[-1].output_size
        self.layers = layers
        print(self.layers)
        self.forward_acc = None  # Stores forward pass result
        self.backward_acc = None  # Stores backward pass result
        if self.sanity_check() == False:
            raise ValueError(f'Sanity check failed, due to incompatible layers: {self.layers}')

    @property
    def output_size(self):
          return self._output_size
    
    def sanity_check(self) -> bool:
        """Check if all layers are compatible in sequence."""
        for i in range(1, len(self.layers)):
            if not self.layers[i].sanity_check(self.layers[i - 1]):
                return False
        return True

    def summary(self) -> List[Dict[str, int]]:
        """Return a summary of each layer."""
        return [layer.summary() for layer in self.layers]

    def clear_accumulations(self) -> None:
        """Clear stored forward and backward results."""
        self.forward_acc = None
        self.backward_acc = None
        for layer in self.layers:
            layer.clear_accumulations()

    def forward(self, x: Any) -> Any:
        """Run the forward pass through all layers."""
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        self.forward_acc = out
        return out

    def backward(self, from_front: Any) -> Any:
        """Run the backward pass through all layers in reverse order."""
        grad = from_front
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        self.backward_acc = grad
        return self.backward_acc