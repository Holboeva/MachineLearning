import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Fayldan ma’lumotni olish
def sklearn_to_df(data_loader):
    X_data = data_loader.data
    X_columns = data_loader.feature_names
    x = pd.DataFrame(X_data, columns=X_columns)

    y_data = data_loader.target
    y = pd.Series(y_data, name='target')

    return x, y

# Ma'lumotni olish
x, y = sklearn_to_df(load_breast_cancer())

# Train-testga bo'lish
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# Custom Logistic Regression class
class CustomLogisticRegression:
    def __init__(self):
        self.weights = None
        self.bias = None
        self.train_accuracies = []
        self.losses = []

    def _transform_x(self, x):
        return np.array(x)

    def _transform_y(self, y):
        return np.array(y)

    def _sigmoid_function(self, x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)

    def _sigmoid(self, x):
        return np.array([self._sigmoid_function(value) for value in x])

    def compute_loss(self, y_true, y_pred):
        y_zero_loss = y_true * np.log(y_pred + 1e-9)
        y_one_loss = (1 - y_true) * np.log(1 - y_pred + 1e-9)
        return -np.mean(y_zero_loss + y_one_loss)

    def compute_gradients(self, x, y_true, y_pred):
        difference = y_pred - y_true
        gradient_b = np.mean(difference)
        gradients_w = np.matmul(x.T, difference) / x.shape[0]
        return gradients_w, gradient_b

    def update_model_parameters(self, error_w, error_b, learning_rate=0.1):
        self.weights -= learning_rate * error_w
        self.bias -= learning_rate * error_b

    def fit(self, x, y, epochs):
        x = self._transform_x(x)
        y = self._transform_y(y)

        self.weights = np.zeros(x.shape[1])
        self.bias = 0

        for i in range(epochs):
            x_dot_weights = np.matmul(self.weights, x.T) + self.bias
            pred = self._sigmoid(x_dot_weights)
            loss = self.compute_loss(y, pred)
            error_w, error_b = self.compute_gradients(x, y, pred)
            self.update_model_parameters(error_w, error_b)

            pred_to_class = [1 if p > 0.5 else 0 for p in pred]
            self.train_accuracies.append(accuracy_score(y, pred_to_class))
            self.losses.append(loss)

    def predict(self, x):
        x = self._transform_x(x)
        x_dot_weights = np.matmul(x, self.weights) + self.bias
        probabilities = self._sigmoid(x_dot_weights)
        return [1 if p > 0.5 else 0 for p in probabilities]

# Modelni chaqiramiz va o'qitamiz
custom_lr = CustomLogisticRegression()
custom_lr.fit(x_train, y_train, epochs=150)

# Bashorat va aniqlik
pred = custom_lr.predict(x_test)
accuracy = accuracy_score(y_test, pred)
print("Custom logistic regression accuracy:", accuracy)

# sklearn model bilan solishtirish
sk_model = LogisticRegression(solver='newton-cg', max_iter=150)
sk_model.fit(x_train, y_train)
pred2 = sk_model.predict(x_test)
accuracy2 = accuracy_score(y_test, pred2)
print("Sklearn logistic regression accuracy:", accuracy2)


import matplotlib.pyplot as plt

# Grafik oynasi (subplot) ochamiz: 1 qatorda 2 ta grafik
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# 1-grafik: Loss (yo‘qotish) grafigi
axs[0].plot(range(1, len(custom_lr.losses) + 1), custom_lr.losses, color='blue')
axs[0].set_title('Loss Curve (Binary Cross-Entropy)', fontsize=14)
axs[0].set_xlabel('Epochs', fontsize=12)
axs[0].set_ylabel('Loss', fontsize=12)
axs[0].grid(True)

# 2-grafik: Accuracy (aniqlik) grafigi
axs[1].plot(range(1, len(custom_lr.train_accuracies) + 1), custom_lr.train_accuracies, color='green')
axs[1].set_title('Accuracy Curve (Training Accuracy)', fontsize=14)
axs[1].set_xlabel('Epochs', fontsize=12)
axs[1].set_ylabel('Accuracy', fontsize=12)
axs[1].grid(True)

# Barcha grafiklarni ko‘rsatamiz
plt.tight_layout()
plt.show()
