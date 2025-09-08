import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, batch_size=32, regularization=0, max_epochs=100, patience=3):
        self.lr = lr
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None

    def fit(self, X, y, batch_size=None, regularization=None, max_epochs=None, patience=None):
        batch_size = batch_size or self.batch_size
        regularization = regularization if regularization is not None else self.regularization
        max_epochs = max_epochs or self.max_epochs
        patience = patience or self.patience

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples, n_features = X.shape
        n_outputs = y.shape[1]

        split = int(n_samples * 0.9)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        train_idx, val_idx = indices[:split], indices[split:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        self.weights = np.zeros((n_features, n_outputs))
        self.bias = np.zeros(n_outputs)
        
        # --- FIX #2: Initialize best_weights here to prevent a None bug ---
        best_weights = self.weights.copy()
        best_bias = self.bias
        # -----------------------------------------------------------------
        
        best_val_loss = float("inf")
        patience_counter = 0
        loss_history = []

        for epoch in range(max_epochs):
            shuffle_idx = np.random.permutation(len(X_train))
            X_train_shuffled, y_train_shuffled = X_train[shuffle_idx], y_train[shuffle_idx]

            for i in range(0, len(X_train_shuffled), batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]

                y_pred = np.dot(X_batch, self.weights) + self.bias
                error = y_pred - y_batch

                grad_w = (2 / len(y_batch)) * (np.dot(X_batch.T, error) + regularization * self.weights)
                grad_b = (2 / len(y_batch)) * np.sum(error, axis=0)

                self.weights -= self.lr * grad_w
                self.bias -= self.lr * grad_b

                batch_loss = np.mean(error ** 2)
                loss_history.append(batch_loss)

            val_pred = self.predict(X_val)
            val_loss = np.mean((val_pred - y_val) ** 2)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = self.weights.copy()
                best_bias = self.bias
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        self.weights = best_weights
        self.bias = best_bias
        
        # --- FIX #1: Ensure this is the last line of the function ---
        return loss_history
        # -------------------------------------------------------------

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)

    def save(self, file_path):
        np.savez(file_path, weights=self.weights, bias=self.bias)
        print(f"Model saved to {file_path}")

    def load(self, file_path):
        data = np.load(file_path, allow_pickle=True)
        self.weights = data["weights"]
        self.bias = data["bias"]
        print(f"Model loaded from {file_path}")