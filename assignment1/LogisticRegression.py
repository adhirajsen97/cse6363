# LogisticRegression.py

import numpy as np

class LogisticRegression:
    """
    Implements a multi-class Logistic Regression (Softmax Regression) model
    using batch gradient descent.
    """
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        
    def _add_intercept(self, X):
        """Adds an intercept term to the input data."""
        return np.c_[np.ones(X.shape[0]), X]

    def _softmax(self, z):
        """Computes the softmax function."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _to_one_hot(self, y, n_classes):
        """Converts an array of labels to a one-hot encoded matrix."""
        return np.eye(n_classes)[y]

    def fit(self, X, y):
        """
        Fits the logistic regression model to the training data.

        Args:
            X (np.ndarray): Input features, shape (n_samples, n_features).
            y (np.ndarray): Target class labels, shape (n_samples,).
        """
        X_b = self._add_intercept(X)
        n_samples, n_features = X_b.shape
        self.n_classes = len(np.unique(y))

        # Initialize weights
        self.weights = np.random.randn(n_features, self.n_classes)

        # One-hot encode the target labels
        y_one_hot = self._to_one_hot(y, self.n_classes)

        # Gradient Descent
        for i in range(self.n_iterations):
            scores = X_b @ self.weights
            probabilities = self._softmax(scores)
            
            error = probabilities - y_one_hot
            gradient = (1 / n_samples) * (X_b.T @ error)
            
            self.weights -= self.learning_rate * gradient

    def predict(self, X):
        """
        Makes class predictions using the trained model.
        """
        if self.weights is None:
            raise RuntimeError("The model has not been trained yet. Call fit() first.")
        
        X_b = self._add_intercept(X)
        scores = X_b @ self.weights
        probabilities = self._softmax(scores)
        return np.argmax(probabilities, axis=1)

    def score(self, X, y):
        """
        Calculates the accuracy of the model.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def save(self, filepath):
        """Saves the model weights to a file."""
        np.save(filepath, self.weights)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Loads model weights from a file."""
        self.weights = np.load(filepath)
        # We need to know the number of classes from the loaded weights
        self.n_classes = self.weights.shape[1]
        print(f"Model loaded from {filepath}")