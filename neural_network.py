import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from typing import List, Callable
from sklearn.model_selection import train_test_split

class Activations:
    @staticmethod
    def sigmoid(Z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-Z))
    
    @staticmethod
    def sigmoid_gradient(Z: np.ndarray) -> np.ndarray:
        s = Activations.sigmoid(Z)
        return s * (1 - s)
    
    @staticmethod
    def tanh(Z: np.ndarray) -> np.ndarray:
        return np.tanh(Z)
    
    @staticmethod
    def tanh_gradient(Z: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(Z)**2
    
    @staticmethod
    def relu(Z: np.ndarray) -> np.ndarray:
        return np.maximum(0, Z)
    
    @staticmethod
    def relu_gradient(Z: np.ndarray) -> np.ndarray:
        return (Z > 0).astype(float)
    
    @staticmethod
    def leaky_relu(Z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(Z > 0, Z, alpha * Z)
    
    @staticmethod
    def leaky_relu_gradient(Z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(Z > 0, 1, alpha)
    
    @staticmethod
    def softmax(Z: np.ndarray) -> np.ndarray:
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

class WeightInitializer:
    @staticmethod
    def zero_init(input_dim: int, output_dim: int) -> tuple:
        W = np.zeros((output_dim, input_dim))
        b = np.zeros((output_dim, 1))
        return W, b
    
    @staticmethod
    def random_init(input_dim: int, output_dim: int) -> tuple:
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        W = np.random.uniform(-scale, scale, (output_dim, input_dim))
        b = np.zeros((output_dim, 1))
        return W, b
    
    @staticmethod
    def normal_init(input_dim: int, output_dim: int) -> tuple:
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        W = np.random.normal(0, scale, (output_dim, input_dim))
        b = np.zeros((output_dim, 1))
        return W, b

class NeuralNetwork:
    def __init__(self, 
                 layer_sizes: List[int],
                 learning_rate: float,
                 activation_function: str,
                 weight_init: str,
                 epochs: int,
                 batch_size: int):
        
        self.layer_sizes = layer_sizes
        self.N = len(layer_sizes)
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Set activation functions
        activation_functions = {
            'sigmoid': (Activations.sigmoid, Activations.sigmoid_gradient),
            'tanh': (Activations.tanh, Activations.tanh_gradient),
            'relu': (Activations.relu, Activations.relu_gradient),
            'leaky_relu': (Activations.leaky_relu, Activations.leaky_relu_gradient)
        }
        self.activation, self.activation_gradient = activation_functions[activation_function]
        
        # Set weight initialization
        weight_inits = {
            'zero': WeightInitializer.zero_init,
            'random': WeightInitializer.random_init,
            'normal': WeightInitializer.normal_init
        }
        self.weight_init = weight_inits[weight_init]
        
        # Initialize parameters
        self.parameters = {}
        self.initialize_parameters()
        
        # For storing training history
        self.train_loss_history = []
        self.val_loss_history = []
        
    def initialize_parameters(self):
        for l in range(1, self.N):
            W, b = self.weight_init(self.layer_sizes[l-1], self.layer_sizes[l])
            self.parameters[f'W{l}'] = W
            self.parameters[f'b{l}'] = b
    
    def forward_propagation(self, X: np.ndarray) -> dict:
        cache = {'A0': X}
        A = X
        
        # Hidden layers
        for l in range(1, self.N-1):
            Z = self.parameters[f'W{l}'].dot(A) + self.parameters[f'b{l}']
            A = self.activation(Z)
            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A
        
        # Output layer (softmax)
        Z = self.parameters[f'W{self.N-1}'].dot(A) + self.parameters[f'b{self.N-1}']
        A = Activations.softmax(Z)
        cache[f'Z{self.N-1}'] = Z
        cache[f'A{self.N-1}'] = A
        
        return cache
    
    def backward_propagation(self, X: np.ndarray, Y: np.ndarray, cache: dict) -> dict:
        m = X.shape[1]
        grads = {}
        
        # Convert Y to one-hot
        Y_one_hot = np.zeros((self.layer_sizes[-1], Y.size))
        Y_one_hot[Y, np.arange(Y.size)] = 1
        
        # Output layer
        dZ = cache[f'A{self.N-1}'] - Y_one_hot
        grads[f'dW{self.N-1}'] = 1/m * dZ.dot(cache[f'A{self.N-2}'].T)
        grads[f'db{self.N-1}'] = 1/m * np.sum(dZ, axis=1, keepdims=True)
        
        # Hidden layers
        for l in reversed(range(1, self.N-1)):
            dA = self.parameters[f'W{l+1}'].T.dot(dZ)
            dZ = dA * self.activation_gradient(cache[f'Z{l}'])
            grads[f'dW{l}'] = 1/m * dZ.dot(cache[f'A{l-1}'].T)
            grads[f'db{l}'] = 1/m * np.sum(dZ, axis=1, keepdims=True)
            
        return grads
    
    def update_parameters(self, grads: dict):
        for l in range(1, self.N):
            self.parameters[f'W{l}'] -= self.lr * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= self.lr * grads[f'db{l}']
    
    def compute_loss(self, Y_pred: np.ndarray, Y: np.ndarray) -> float:
        m = Y.size
        Y_one_hot = np.zeros((self.layer_sizes[-1], Y.size))
        Y_one_hot[Y, np.arange(Y.size)] = 1
        log_probs = np.multiply(Y_one_hot, np.log(Y_pred + 1e-8))
        return -np.sum(log_probs) / m
    
    def fit(self, X: np.ndarray, Y: np.ndarray, X_val: np.ndarray = None, Y_val: np.ndarray = None):
        m = X.shape[1]
        
        for epoch in range(self.epochs):
            # Mini-batch training
            for i in range(0, m, self.batch_size):
                batch_X = X[:, i:min(i+self.batch_size, m)]
                batch_Y = Y[i:min(i+self.batch_size, m)]
                
                # Forward propagation
                cache = self.forward_propagation(batch_X)
                
                # Backward propagation
                grads = self.backward_propagation(batch_X, batch_Y, cache)
                
                # Update parameters
                self.update_parameters(grads)
            
            # Compute training loss
            cache = self.forward_propagation(X)
            train_loss = self.compute_loss(cache[f'A{self.N-1}'], Y)
            self.train_loss_history.append(train_loss)
            
            # Compute validation loss
            if X_val is not None and Y_val is not None:
                val_cache = self.forward_propagation(X_val)
                val_loss = self.compute_loss(val_cache[f'A{self.N-1}'], Y_val)
                self.val_loss_history.append(val_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.epochs} - Train Loss: {train_loss:.4f}", 
                      f"- Val Loss: {val_loss:.4f}" if X_val is not None else "")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        cache = self.forward_propagation(X)
        return np.argmax(cache[f'A{self.N-1}'], axis=0)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        cache = self.forward_propagation(X)
        return cache[f'A{self.N-1}']
    
    def score(self, X: np.ndarray, Y: np.ndarray) -> float:
        predictions = self.predict(X)
        return np.mean(predictions == Y)

def load_and_preprocess_mnist(data_path: str):
    # Load data
    data = pd.read_csv(data_path)
    data = np.array(data)
    
    # Split features and labels
    Y = data[:, 0]
    X = data[:, 1:] / 255.0  # Normalize pixel values
    
    # Reshape X to (784, m) format
    X = X.T
    
    # Split into train, validation, and test sets (80:10:10)
    X_train, X_temp, Y_train, Y_temp = train_test_split(X.T, Y, test_size=0.2, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)
    
    return (X_train.T, Y_train), (X_val.T, Y_val), (X_test.T, Y_test)

def train_and_save_models(data_path: str):
    # Load and preprocess data
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = load_and_preprocess_mnist(data_path)
    
    # Network configuration
    input_size = 784
    hidden_layers = [256, 128, 64, 32]
    output_size = 10
    layer_sizes = [input_size] + hidden_layers + [output_size]
    
    # Training parameters
    epochs = 100
    batch_size = 128
    learning_rate = 2e-5
    
    activations = ['sigmoid', 'tanh', 'relu', 'leaky_relu']
    weight_inits = ['zero', 'random', 'normal']
    
    # Train models with different configurations
    for activation in activations:
        for weight_init in weight_inits:
            print(f"\nTraining model with {activation} activation and {weight_init} initialization")
            
            # Initialize and train model
            model = NeuralNetwork(
                layer_sizes=layer_sizes,
                learning_rate=learning_rate,
                activation_function=activation,
                weight_init=weight_init,
                epochs=epochs,
                batch_size=batch_size
            )
            
            model.fit(X_train, Y_train, X_val, Y_val)
            
            # Save model
            filename = f'mnist_model_{activation}_{weight_init}.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            
            # Plot learning curves
            plt.figure(figsize=(10, 5))
            plt.plot(model.train_loss_history, label='Training Loss')
            plt.plot(model.val_loss_history, label='Validation Loss')
            plt.title(f'Learning Curves ({activation} activation, {weight_init} init)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f'learning_curves_{activation}_{weight_init}.png')
            plt.close()
            
            # Print test accuracy
            test_accuracy = model.score(X_test, Y_test)
            print(f"Test accuracy: {test_accuracy:.4f}")

# Example usage
if __name__ == "__main__":
    data_path = "Section B\\mnist_train.csv"  # Update with your MNIST data path
    train_and_save_models(data_path)