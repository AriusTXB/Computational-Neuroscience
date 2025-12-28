import numpy as np

class ReservoirNetwork:
    def __init__(self, n_res=300, spectral_radius=0.95, sparsity=0.2):
        self.n_res = n_res
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.W_in = None
        self.W_res = None
        self.W_out = None

    def initialize(self, seed=42):
        np.random.seed(seed)
        self.W_in = (np.random.rand(self.n_res, 1) - 0.5) * 2.0
        self.W_res = np.random.rand(self.n_res, self.n_res) - 0.5
        mask = np.random.rand(self.n_res, self.n_res) > self.sparsity
        self.W_res[mask] = 0
        rho = np.max(np.abs(np.linalg.eigvals(self.W_res)))
        self.W_res *= (self.spectral_radius / rho)

    def train(self, u_train, target, washout=100, reg=1e-8):
        T = len(u_train)
        states = np.zeros((self.n_res, T))
        x = np.zeros((self.n_res, 1))
        
        for t in range(T):
            x = np.tanh(np.dot(self.W_in, u_train[t]) + np.dot(self.W_res, x))
            states[:, t] = x[:, 0]
            
        X = states[:, washout:].T
        Y = target[washout:]
        
        # Ridge Regression
        X_T = X.T
        self.W_out = np.dot(np.dot(np.linalg.inv(np.dot(X_T, X) + reg * np.eye(self.n_res)), X_T), Y)
        return states

    def predict(self, u_test):
        T = len(u_test)
        states = np.zeros((self.n_res, T))
        x = np.zeros((self.n_res, 1)) # Should carry over state in real app
        
        for t in range(T):
            x = np.tanh(np.dot(self.W_in, u_test[t]) + np.dot(self.W_res, x))
            states[:, t] = x[:, 0]
            
        return np.dot(states.T, self.W_out)