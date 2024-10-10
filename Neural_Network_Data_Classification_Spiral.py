import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Function to generate spiral-shaped data points
def generate_spiral_data(num_points, noise=0.3):
    n = np.sqrt(np.random.rand(num_points, 1)) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(num_points, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(num_points, 1) * noise
    return np.hstack((d1x, d1y))

# Generate data for two classes
num_data_points = 100
x0 = generate_spiral_data(num_data_points)  # Data for class 0
y0 = np.zeros(num_data_points)              # Labels for class 0
x1 = generate_spiral_data(num_data_points)  # Data for class 1
y1 = np.ones(num_data_points)               # Labels for class 1

# Combine and convert data into PyTorch tensors
x = np.vstack((x0, x1))
y = np.hstack((y0, y1))
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Define a neural network model for binary classification
class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(NeuralNetwork, self).__init__()
        self.hidden_1 = torch.nn.Linear(input_size, hidden_sizes[0])
        self.hidden_2 = torch.nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.hidden_3 = torch.nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.dropout = torch.nn.Dropout(p=0.5)
        self.output = torch.nn.Linear(hidden_sizes[2], output_size)

    def forward(self, x):
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = self.dropout(F.relu(self.hidden_3(x)))  # Adding dropout for regularization
        return self.output(x)

# Model configuration
input_size = 2  # Each data point has two features (x, y)
hidden_sizes = [32, 64, 128]  # Increased sizes of the hidden layers
output_size = 2  # Binary classification (two classes)
model = NeuralNetwork(input_size, hidden_sizes, output_size)

# Training configuration
learning_rate = 0.001  # Lower learning rate for finer adjustments
num_epochs = 30000     # Increased number of epochs
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Using Adam optimizer
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)  # Reduce LR every 10k epochs
loss_function = torch.nn.CrossEntropyLoss()

# Enable interactive plotting
plt.ion()
plt.figure(figsize=(15, 10))
plt.get_current_fig_manager().full_screen_toggle()


# Training loop
for epoch in range(num_epochs):
    # Perform a forward pass and calculate loss
    output = model(x)
    loss = loss_function(output, y)

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    # Optional: Gradient clipping to improve stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
    optimizer.step()
    # Step the learning rate scheduler
    scheduler.step()

    # Plot the decision boundary and data every few epochs
    if epoch % 200 == 0:
        plt.cla()
        predictions = torch.max(output, 1)[1]
        predicted_labels = predictions.numpy()
        true_labels = y.numpy()

        # Generate a mesh grid for plotting the decision boundary
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
        
        # Predict class for each point in the grid
        with torch.no_grad():
            grid_output = model(grid)
            grid_predictions = torch.max(grid_output, 1)[1].numpy().reshape(xx.shape)

        # Plot the decision boundary
        plt.contourf(xx, yy, grid_predictions, alpha=0.5, cmap='coolwarm')

        # Plot data points with white edges
        plt.scatter(x[y == 0, 0].numpy(), x[y == 0, 1].numpy(), c='blue', edgecolors='white', label='Class 0', s=40)
        plt.scatter(x[y == 1, 0].numpy(), x[y == 1, 1].numpy(), c='red', edgecolors='white', label='Class 1', s=40)
        
        # Plot legend
        plt.legend(loc='upper right', fontsize=12, framealpha=0.8)

        # Display accuracy
        accuracy = (predicted_labels == true_labels).mean()
        plt.text(x_max - 5, y_min + 1, f'Accuracy = {accuracy:.2f}', fontdict={'size': 20, 'color': 'green'})
        plt.title(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}')
        plt.pause(0.1)

# Disable interactive mode and display the final plot
plt.ioff()
plt.show()
