# numpy-neural-network
A hands-on project aimed at understanding the intricacies of Artificial Neural Networks by building one from scratch using Numpy.

## Introduction
In the era of deep learning, understanding the foundational building blocks of neural networks is crucial. This project serves as an educational tool to dive deep into the workings of neural networks by building one from the ground up using just Numpy.

# Project structure

```
numpy-neural-network
├── LICENSE                   # License file for the project
├── notebook
│   ├── experiments.ipynb     # Jupyter notebook for various ANN experiments and visualizations
│   └── training.ipynb        # Jupyter notebook dedicated for training the ANN model
├── README.md                 # Documentation and overview of the project
├── requirements.txt          # List of Python libraries required for this project
└── src
    ├── __init__.py           # Initialization file for the Python package
    ├── model.py              # Contains the main ANN model implementation
    └── utils
        ├── activate.py       # Activation functions like sigmoid, softmax, etc.
        ├── gradients.py      # Functions related to gradient computations
        ├── __init__.py       # Initialization file for the utils package
        └── loss.py           # Loss functions like cross-entropy, mean squared error, etc.
```

# Environment

## Python
```bash
$ python --version
Python 3.9.4
```

## install dependecy
```bash
$ pip install -r requirements.txt
```

# Usage
## Training the Model
Navigate to the `notebook` directory and open `training.ipynb`. This notebook provides a step-by-step guide to train the ANN model. Alternatively, you can train the model using the following code:

```python
from src.model import ANN


model = ANN(4, 5, 3)
x, y = <load your data>
model.fit(x, y, epochs=1000)
```

## Making Predictions
Once the model is trained, you can use the `predict` method from the `ANN` class in `model.py` to make predictions on new data:

```python
predicted_y = model.predict(x) # making predictions
accuracy_score(y, predicted_y)  # Caculating accuracy score
```

## Comprehensive Guide
For a more detailed walkthrough on using the ANN model, refer to the steps provided in `notebook/training.ipynb`.