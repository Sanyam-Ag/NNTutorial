# OLS closed form method: 
# w = Inverse(Transpose(X)*X)*Transpose(X)*y
# b = y_mean - sum(respective_weight * repective_X_mean)

# if large dataset use SGDRegressor

import numpy as np

class LinearReg:

    def __init__(self, fit_intercept = True):
        self.fit_intercept = fit_intercept
        self.nweights = 0   # number of weights
        self._weights = []  # empty weights list
        self._bias = 0

    def __repr__(self):
        return "LinearRegression()"
    
    def fit(self, Xtrain, ytrain):
        if Xtrain is None or ytrain is None: raise ValueError("Insufficient parameters passed")
        elif not isinstance(Xtrain, (list, tuple, np.ndarray)) or not isinstance(Xtrain, (list, tuple, np.ndarray)): raise ValueError("Invalid datatype of dataset")
        elif (len(Xtrain) == 0 or len(ytrain) == 0): raise ValueError("Empty data passed")
        elif(len(Xtrain) != len(ytrain)): raise ValueError("Unmatched X and Y data points")

        Xtrain = np.asarray(Xtrain)
        ytrain = np.asarray(ytrain)
    
        # _fit_helper
        self._fit_helper(Xtrain, ytrain)

        print("Model trained!")

    def _fit_helper(self, Xtrain, ytrain):
        self.nweights = len(Xtrain)

        ones = np.ones((Xtrain.shape[0], 1))
        X_design = np.hstack((ones, Xtrain))    # added ones for calculation of bias alongwith weights

        # (X^T @ X)^(-1) @ X^T @ y
        X_T = X_design.T
        w = np.linalg.inv(X_T @ X_design) @ X_T @ ytrain

        self._weights = w[1:]
        if self.fit_intercept:
            self._bias = w[0]

    def predict(self, Xfeaturesval):
        return (np.asarray(Xfeaturesval) @ self._weights) + self._bias
    
    def score(self, Xtest, ytest):
        ytest = np.asarray(ytest)

        y_pred = self.predict(Xtest)
        y_mean = np.mean(ytest)

        #  SS_res and SS_tot
        ss_res = np.sum((ytest - y_pred) ** 2)
        ss_tot = np.sum((ytest - y_mean) ** 2)

        # R^2 score
        return (1 - (ss_res / ss_tot))