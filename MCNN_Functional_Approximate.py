import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# True parameters of the distribution
true_mean = 4
true_std = 2

# Number of samples
num_samples = 1500

# Generate observed data from the true distribution
observed_data = np.random.normal(true_mean, true_std, num_samples)
print(observed_data)

# Convert observed data to PyTorch tensor
observed_data_tensor = torch.tensor(observed_data, dtype=torch.float32).view(-1, 1)

# Define a simple neural network model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 15)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(15, 15)
        self.tanh = nn.Tanh()
        self.fc3 = nn.Linear(15, 15)
        self.tanh = nn.Tanh()
        self.fc4 = nn.Linear(15, 2) # 2 output nodes for mean and std

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        x = self.tanh(x)
        x = self.fc4(x)
        return x

# Instantiate the model, loss function, and optimizer
model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    output = model(observed_data_tensor)
    
    # Compute the loss (MSE loss between predicted mean/std and true mean/std)
    loss = criterion(output, torch.tensor([true_mean, true_std], dtype=torch.float32).view(1, -1))
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Generate samples from the neural network
generated_samples = model(observed_data_tensor).detach().numpy()
print(generated_samples)

# Plot the results
plt.figure(figsize=(12, 6))

# Plot the true distribution
plt.subplot(1, 2, 1)
plt.title('True Distribution')
plt.hist(observed_data, bins=50, density=True, color='blue', alpha=0.6)
plt.axvline(true_mean, color='red', linestyle='dashed', linewidth=2, label='True Mean')
plt.axvline(true_mean + true_std, color='green', linestyle='dashed', linewidth=2, label='True Mean + Std')
plt.legend()

# Plot the generated distribution
plt.subplot(1, 2, 2)
plt.title('Generated Distribution')
plt.hist(generated_samples[:, 0], bins=30, density=True, color='orange', alpha=0.6)
# plt.axvline(true_mean, color='red', linestyle='dashed', linewidth=2, label='True Mean')
# plt.axvline(true_mean + true_std, color='green', linestyle='dashed', linewidth=2, label='True Mean + Std')
plt.legend()

plt.show()
