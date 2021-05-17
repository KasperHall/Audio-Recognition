import numpy as np
import scipy.sparse as sparse 

class SingleReadoutLayer():
    """
    Predicts the most likely output class from a vector of inputs through ridge regression
    """
    def __init__(self, n_classes: int, n_features: int) -> None:
        self.output_weights = np.zeros((n_features, n_classes))
        self.n_classes = n_classes
        self.n_features = n_features

    def predict(self, x):
        x = x.numpy().flatten()
        return np.argmax(x @ self.output_weights)

    def test(self, test_set):
        correct = 0
        for x, target in test_set:
            if self.predict(x) == target:
                correct += 1
                
        return correct/len(test_set)

    def train(self, training_set, n_steps, n_freq):

        n_samples = len(training_set)
        
        design_matrix = np.zeros((n_samples, n_steps*n_freq))
        target_output = np.zeros((n_samples, self.n_classes))

        for i, (x, target) in enumerate(training_set):
            design_matrix[i, :] = x.numpy().flatten()
            target_output[i, target] = 1

        self.output_weights[:, :] = np.linalg.inv(design_matrix.T @ design_matrix + 0.1*np.eye(n_steps*n_freq)) @ design_matrix.T @ target_output

class MultiReadoutLayer():
    """
    Predicts the most likely output class from a series of vectors through ridge regression
    """

    def __init__(self, n_classes: int, n_features: int, n_steps: int) -> None:
        self.readout_layers = [SingleReadoutLayer(n_classes, n_features) for i in range(n_steps)]
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_steps

    def __call__(self, x):
        result = np.array([x_t @ l.output_weights for l, x_t in zip(self.readout_layers, x)])
        return np.argmax(np.mean(result, 1))
    
    def test(self, test_set):
        correct = 0
        for x, target in test_set:
            if self.predict(x) == target:
                correct += 1
                
        return correct/len(test_set)
    
    def train(self, training_set, n_steps, n_freq):
        for t, l in enumerate(self.n_steps):
            training_set = [(x[t], label_id) for x, label_id in training_set]
            l.train(training_set)
            print(f"{t}/{self.n_steps} trained.")

class ReservoirLayer():
    """
    Simple Echo state network layer
    """
    def __init__(self, reservoir_size: int, input_dim: int) -> None:
        self.reservoir_size = reservoir_size

        self.input_weights = -0.1 + 0.2*np.random.rand(reservoir_size, input_dim)
        self.reservoir_weights = sparse.random(reservoir_size, reservoir_size, density=0.01, data_rvs = lambda shape: -1 + 2*np.random.rand(shape))
        self.reservoir_weights = self.reservoir_weights/(np.max(np.real(np.linalg.eigvals(self.reservoir_weights))))
        self.reservoir_bias = -0.1 + 0.2*np.random.rand(reservoir_size, 1)

    def __call__(self, x, return_all_states = False):
        reservoir_state = np.zeros((self.reservoir_size, 1))
        
        reservoir_state = np.array([[]])
        for x_t in x:
            result= np.tanh(self.input_weights @ x_t + self.reservoir_weights @ reservoir_state + self.reservoir_bias)
            if return_all_states:
                reservoir_state = np.append(reservoir_state, result, axis=-1)
            else:
                reservoir_state = result
    
        return reservoir_state