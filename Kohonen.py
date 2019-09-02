import numpy as np

class KohonenNN:
    def __init__(self, in_units, K):
        self.W = np.random.randn(in_units, K)

    def forward(self, x):
        E = np.sum((self.W-x)**2, axis=0).reshape(-1, 1)
        return E

    def train(self, data, learning_rate, R, R_decay=0.5, lr_decay= 0.5, epochs=1):
        for epoch in epochs:
            for x in data:
                x = x.reshape((-1, 1))
                E = self.forward(x)
                indices = E.argsort(axis=0)[E.shape[0]-R:, 0]
                self.W[indices] = self.W[indices] + learning_rate*(x[indices]-self.W[indices])
            R = int(np.ceil(R*R_decay))
            learning_rate *= lr_decay
