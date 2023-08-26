import pickle

import numpy as np

from .utils.activate import sigmoid_function, softmax_function
from .utils.gradients import dfunc
from .utils.loss import cross_entropy_err


class ANN:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid_function(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax_function(a2)
        return y

    def loss(self, x, y):
        y_hat = self.predict(x)
        return cross_entropy_err(y_hat, y)

    def numerical_gradient(self, x, y):
        loss_W = lambda W: self.loss(x, y)
        grads = {}
        grads["W1"] = dfunc(loss_W, self.params["W1"])
        grads["b1"] = dfunc(loss_W, self.params["b1"])
        grads["W2"] = dfunc(loss_W, self.params["W2"])
        grads["b2"] = dfunc(loss_W, self.params["b2"])
        return grads

    def fit(self, x, y, epochs=3000, lr=0.01):
        train_loss = []
        for i in range(epochs):
            grad = self.numerical_gradient(x, y)
            for key in ("W1", "b1", "W2", "b2"):
                self.params[key] = self.params[key] - lr * grad[key]
            loss = self.loss(x, y)
            train_loss.append(loss)
            # 每當完成 10% 的訓練 epoch 時，輸出一條日誌
            if i % (epochs // 10) == 0:
                print(
                    f"Training progress: {i/epochs*100+10:.2f}% completed. Current loss: {self.loss(x, y):.4f}"
                )

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
