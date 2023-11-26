import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Function to generate samples from the true bivariate normal distribution
def generate_true_samples(num_samples):
    mean = [0, 0]
    covariance_matrix = [[1, 0.5], [0.5, 1]]
    return np.random.multivariate_normal(mean, covariance_matrix, num_samples)

# Marginalise Distribution PDF
def custom_function(v1,v2,v3):
    result_tensor = (v3 / v1) ** (v2 - 1) * torch.exp(-v3)
    return result_tensor

# Neural network model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)  # 2 output nodes for bivariate normal distribution
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

# Generate true samples
num_true_samples = 1000
true_samples = generate_true_samples(num_true_samples)

# Convert true samples to PyTorch tensor
true_samples_tensor = torch.tensor(true_samples, dtype=torch.float32)

# Create vector variables
v1 = true_samples_tensor[:,0]
v2 = true_samples_tensor[:,1] 
observed_data = np.random.normal(0, 1, num_true_samples) #Taking True Mean and True Variance of x-coordinate
v3 = torch.tensor(observed_data, dtype=torch.float32)

# Marginal Distribution PDF
v4 = custom_function(v1,v2,v3)

# Data cleaning of 'nan' entry
simulated_tensor = v4[~torch.isnan(v4)]

# Create true sample tensor
num = simulated_tensor.numel()
true_samples = np.random.normal(0, 1, num)
true_samples_tensor = torch.tensor(true_samples , dtype=torch.float32)

# Instantiate the model, loss function, and optimizer
model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    output = model(simulated_tensor.view(-1, 1))  # Using the x-coordinate
    
    # Compute the loss (MSE loss between predicted mean and true mean)
    loss = criterion(simulated_tensor, true_samples_tensor.view(-1, 1))
    
    # Backward pass and optimization
    optimizer.zero_grad()
    # loss.backward()
    optimizer.step()
    
# Generate samples from the neural network
generated_samples = model(simulated_tensor.view(-1, 1)).detach().numpy()
generated_samples_1 = generated_samples.reshape(-1, 1)
num2 = generated_samples_1.size

# Clip generated sample array
upper_bound = 5
lower_bound = -5
generated_samples_2 = np.clip(generated_samples_1, lower_bound, upper_bound)

# Generate x values for PDF plot
x_values = np.linspace(lower_bound, upper_bound, num2)

# Calculate the PDF values
pdf_values = norm.pdf(x_values, 0, 1)

# Figure constrint
fig, ax = plt.subplots(figsize=(8, 4))

# Plot the scatter plot of samples
plt.scatter(generated_samples_2, np.zeros_like(generated_samples_2), label='Samples', alpha=0.7)

# Plot the normal distribution PDF
plt.plot(x_values, pdf_values, label='Normal Distribution PDF', color='orange')

# Customize the plot
plt.title('Samples and Normal Distribution PDF')
plt.xlabel('x')
plt.ylabel('Probability Density Function (PDF)')
plt.legend()
plt.show()

