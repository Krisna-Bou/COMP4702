import numpy as np

# XOR input and expected output
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([[0], [1], [1], [0]])

# Sigmoid and derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Matrix multiplication with loops
def matmul(A, B):
    result = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                result[i][j] += A[i][k] * B[k][j]
    return result

# Xavier Initialization
def xavier_init(fan_in, fan_out):
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

# Initialize weights and biases
np.random.seed(42)
input_size = 2
hidden_size = 2
output_size = 1

W1 = xavier_init(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = xavier_init(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Training loop
epochs = 20000
learning_rate = 0.5  # Higher LR works better with Xavier

for epoch in range(epochs):
    # Forward pass
    z1 = matmul(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = matmul(a1, W2) + b2
    a2 = sigmoid(z2)

    # Loss (MSE)
    loss = np.mean((a2 - y) ** 2)

    # Backpropagation
    d_a2 = 2 * (a2 - y)
    d_z2 = d_a2 * sigmoid_derivative(z2)

    dW2 = matmul(a1.T, d_z2)
    db2 = np.sum(d_z2, axis=0, keepdims=True)

    d_a1 = matmul(d_z2, W2.T)
    d_z1 = d_a1 * sigmoid_derivative(z1)

    dW1 = matmul(X.T, d_z1)
    db1 = np.sum(d_z1, axis=0, keepdims=True)

    # Gradient Descent Update
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # Print progress
    if epoch % 2000 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

# Final predictions
print("\n--- Final predictions ---")
z1 = matmul(X, W1) + b1
a1 = sigmoid(z1)
z2 = matmul(a1, W2) + b2
a2 = sigmoid(z2)

print("Raw outputs:")
print(a2)

print("\nRounded predictions (should be [0, 1, 1, 0]):")
print(np.round(a2))
