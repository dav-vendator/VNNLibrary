# VNNLibrary

A custom-built Python library for implementing neural networks, including fully connected layers, activation functions like sigmoid, and popular architectures like CNN, ResNet, and RNN. This library is designed with extensibility in mind, allowing users to build and experiment with new layers and models. Objective behind building it is *learning*.

## Features

- **Core Layers**: Custom implementations of linear layers (`VLinearLayer`), activation functions like `VSigmoid`, and more.
- **Flexible Model Architecture**: Supports constructing custom models including CNNs, ResNets, and RNNs.
- **Extensible Framework**: You can easily add new layers, models, or features to the library.
- **Customizable Learning**: Define forward and backward passes for each component to fine-tune the learning process.
- **Easy Integration**: Compatible with `NumPy` for fast computations.

## Installation

### Clone the Repository
```bash
git clone https://github.com/dav-vendator/VNNLibrary.git
cd VNNLibrary
```

### Install Requirements
The library relies on `NumPy` for matrix computations. Install it via `pip`:
```bash
pip install -r requirements.txt
```

### Directory Structure
```
VNNLibrary/
│
├── src/
│   ├── layers/
│   │   ├── __init__.py            # Initializes the layers submodule
│   │   ├── vlinear_layer.py       # Implementation of linear layers
│   │   └── vsigmoid_layer.py      # Implementation of sigmoid activation
│   ├── models/
│   │    ├── __init__.py            # Initializes the models submodule
│   │    ├── vsimple_cnn.py         # Simple CNN architecture
│   │    ├── vresnet.py             # ResNet architecture
│   │    └── vrnn.py                # RNN architecture
│   └── Orchestrators/
│       ├── __init__.py            # Initializes the orchestrators submodule
│       └── vsequence.py           # Simple feedforward orchestrator
├── tests/                         # Unit tests for layers and models
│
├── examples/                      # Usage examples and scripts to train models
│
├── requirements.txt               # List of dependencies
├── README.md                      # Documentation
└── LICENSE                        # License information
```

## Usage

### 1. **Creating a Custom Model**
```python
import numpy as np
from src.layers.vlinear_layer import VLinearLayer
from src.layers.vsigmoid_layer import VSigmoidLayer
from src.orchestrators import VSequence

# Example: A simple feedforward network with one hidden layer
simpleNetwork = VSequence(
   name = 'A simple Network',
   VLinearLayer(input_size=3, output_size=5),
   VSigmoidLayer(),
   VLinearLayer(input_size=5, output_size=1)
)

input_data = np.random.randn(3)
output = simpleNetwork(input_data)
print(f"Model output: {output}")
```

### 2. **Training a Model**
Training can be done by implementing a custom loop that calls `forward()` and `backward()` for each batch.

### 3. **Adding Dropout Layer**
You can easily extend the library by adding custom layers like Dropout. Simply add a new file in `layers/`, and update the imports in `__init__.py`.

## Roadmap

### Planned Features
- **Dropout and Batch Normalization**: Implement commonly used regularization techniques.
- **Training and Optimization Modules**: Implement optimizers such as SGD, Adam, and learning rate schedulers.
- **Pre-built Models**: CNN, ResNet, and RNN architectures available out-of-the-box.
- **GPU Acceleration**: Use `CuPy` or similar libraries to allow for GPU-based computation.
- **Support for PyTorch and TensorFlow Interoperability**: Convert models between frameworks for flexibility.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for new features, bug fixes, or documentation updates.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.