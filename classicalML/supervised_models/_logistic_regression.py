# Logistic Regression using Gradient Descent
# sigmoid: σ(z) = 1 / (1 + exp(-z))   maps any real z → (0, 1)
# Loss (Binary Cross-Entropy): L = -1/n * Σ [ y*log(ŷ) + (1-y)*log(1-ŷ) ]
# Gradients: dL/dw = (1/n) * X^T @ (ŷ - y)
#            dL/db = (1/n) * Σ(ŷ - y)
# Weight update: w := w - α * dL/dw  |  b := b - α * dL/db
# For large datasets use SGDClassifier (mini-batch gradient descent)

import numpy as np

class LogisticReg:
    def __init__(self, learning_rate=0.01, n_iterations=1000, fit_intercept=True, threshold=0.5):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.threshold = threshold
        self.nweights = 0       # number of features
        self._weights = []      # empty weights list
        self._bias = 0
        self._losses = []       # cross-entropy loss per iteration

    def __repr__(self):
        return "LogisticRegression()"

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))    # clipped to avoid overflow

    def fit(self, Xtrain, ytrain):
        if Xtrain is None or ytrain is None: raise ValueError("Insufficient parameters passed")
        elif not isinstance(Xtrain, (list, tuple, np.ndarray)) or not isinstance(ytrain, (list, tuple, np.ndarray)): raise ValueError("Invalid datatype of dataset")
        elif (len(Xtrain) == 0 or len(ytrain) == 0): raise ValueError("Empty data passed")
        elif (len(Xtrain) != len(ytrain)): raise ValueError("Unmatched X and Y data points")
        Xtrain = np.asarray(Xtrain, dtype=float)
        ytrain = np.asarray(ytrain, dtype=float)
        if Xtrain.ndim == 1:
            Xtrain = Xtrain.reshape(-1, 1)  # ensure 2-D input

        # _fit_helper
        self._fit_helper(Xtrain, ytrain)
        print("Model trained!")

    def _fit_helper(self, Xtrain, ytrain):
        n_samples, n_features = Xtrain.shape
        self.nweights = n_features
        self._weights = np.zeros(n_features)
        self._bias = 0
        self._losses = []

        for _ in range(self.n_iterations):
            # forward pass
            z = Xtrain @ self._weights + self._bias
            y_pred = self._sigmoid(z)

            # gradients
            error = y_pred - ytrain
            dw = (1 / n_samples) * (Xtrain.T @ error)
            db = (1 / n_samples) * np.sum(error)

            # update
            self._weights -= self.learning_rate * dw
            if self.fit_intercept:
                self._bias -= self.learning_rate * db

            # binary cross-entropy loss  (ε = 1e-9 prevents log(0))
            loss = -np.mean(
                ytrain * np.log(y_pred + 1e-9) +
                (1 - ytrain) * np.log(1 - y_pred + 1e-9)
            )
            self._losses.append(loss)

    def predict_proba(self, Xfeaturesval):
        """Return probability estimates P(y = 1)."""
        X = np.asarray(Xfeaturesval, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self._sigmoid(X @ self._weights + self._bias)

    def predict(self, Xfeaturesval):
        """Return binary class labels using self.threshold."""
        return (self.predict_proba(Xfeaturesval) >= self.threshold).astype(int)

    def score(self, Xtest, ytest):
        """Accuracy: fraction of correctly classified samples."""
        ytest = np.asarray(ytest)
        y_pred = self.predict(Xtest)
        return np.mean(y_pred == ytest)