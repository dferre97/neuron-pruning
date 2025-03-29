"""
This script compares three approaches on Fashion MNIST:
1. Neuron-level Popup Model: Prunes entire neurons using a score-based selection method inspired by edge-popup.
2. Weight-level Popup Model (Original Edge-Popup): Prunes individual weights using score-based selection.
3. Standard MLP: A baseline fully connected network trained normally.

All models use the same architecture (3 layers: 2 hidden layers with the same number of neurons).
Training metrics (training and test accuracy) are recorded and plotted.
The plots are saved in a folder named results_2hl/{day-hour-minutes-seconds}.
"""

import logging
import math
import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

# ----------------------------
# Utility: Straight-through estimator for subset selection
# ----------------------------
class GetNeuronSubset(Function):
    @staticmethod
    def forward(ctx, scores, k):
        """Selects the top-k% neurons based on their scores."""
        out = scores.clone()
        _, idx = scores.sort()
        j = int((1 - k) * scores.numel())
        out[idx[:j]] = 0
        out[idx[j:]] = 1
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """Straight-through estimator: Pass gradients as-is."""
        return grad_output, None

class GetWeightSubset(Function):
    @staticmethod
    def forward(ctx, scores, k):
        """Selects the top-k% weights based on their scores."""
        out = scores.clone()
        scores_flat = scores.view(-1)
        _, idx = scores_flat.sort()
        j = int((1 - k) * scores_flat.numel())
        scores_flat[idx[:j]] = 0
        scores_flat[idx[j:]] = 1
        out = scores_flat.view_as(scores)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """Straight-through estimator: Pass gradients as-is."""
        return grad_output, None

# ----------------------------
# Model Definitions
# ----------------------------
# Neuron-level Popup Model: Prunes entire neurons.
class NeuronPopupLayer(nn.Module):
    def __init__(self, in_features, out_features, k=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k  # Fraction of neurons to keep
        
        # Randomly initialize weights and biases, and freeze them.
        # self.weights = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.weights = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        # Only the scores are trainable.
        self.scores = nn.Parameter(torch.ones(out_features))

        # Initialize weights and scores using Kaiming Normal and Uniform
        nn.init.kaiming_normal_(self.weights, mode="fan_in", nonlinearity='relu')
        
        # # Initialize scores using a uniform distribution with bounds computed from fan_in.
        # # For a uniform distribution U(-a, a), variance = a^2/3, and we want variance 2/in_features (for ReLU, see Kaiming's paper).
        # # So, a = sqrt(6/in_features)
        # # nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5)) # doesn't work because of tensor shape, try to find good initialization for neuron scores..
        # bound = math.sqrt(6 / in_features)
        # self.scores = nn.Parameter(torch.empty(out_features))
        # nn.init.uniform_(self.scores, -bound, bound)

    def forward(self, x):
        # Get the neuron subset mask
        neuron_mask = GetNeuronSubset.apply(self.scores.abs(), self.k)
        pruned_weights = self.weights * neuron_mask[:, None]  # Mask applied per neuron
        pruned_bias = self.bias * neuron_mask  # Mask applied to biases
        return F.linear(x, pruned_weights, pruned_bias)

class NeuronPopupNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, k=0.5):
        super().__init__()
        self.fc1 = NeuronPopupLayer(input_dim, hidden_dim, k)
        self.fc2 = NeuronPopupLayer(hidden_dim, hidden_dim, k)  # Added second hidden layer
        self.fc3 = NeuronPopupLayer(hidden_dim, output_dim, k)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  # Forward pass through second hidden layer
        x = self.fc3(x)
        return x

# Weight-level Popup Model: Prunes individual weights.
class WeightPopupLayer(nn.Module):
    def __init__(self, in_features, out_features, k=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        
        # Randomly initialize weights and biases, and freeze them.
        self.weights = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        # Only the scores are trainable.
        self.scores = nn.Parameter(torch.ones(out_features, in_features))
        
        # Initialize weights and scores using Kaiming Normal
        nn.init.kaiming_normal_(self.weights, mode="fan_in", nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def forward(self, x):
        weight_mask = GetWeightSubset.apply(self.scores.abs(), self.k)
        pruned_weights = self.weights * weight_mask
        return F.linear(x, pruned_weights, self.bias)

class WeightPopupNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, k=0.5):
        super().__init__()
        self.fc1 = WeightPopupLayer(input_dim, hidden_dim, k)
        self.fc2 = WeightPopupLayer(hidden_dim, hidden_dim, k)  # Added second hidden layer
        self.fc3 = WeightPopupLayer(hidden_dim, output_dim, k)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  # Forward pass through second hidden layer
        x = self.fc3(x)
        return x

# Standard MLP: A normal fully connected network.
class StandardMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Added second hidden layer
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  # Forward pass through second hidden layer
        x = self.fc3(x)
        return x

# ----------------------------
# Training and Evaluation Functions
# ----------------------------
def train(model, device, train_loader, optimizer, epoch):
    # Train the model
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    logger.info(f"Epoch {epoch}: Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    accuracy = 100. * correct / total
    return accuracy

# ----------------------------
# Main Experiment Setup
# ----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Data preparation: Fashion MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # ----------------------------
    # Save configuration details
    # ----------------------------
    config_path = os.path.join(results_dir, 'config.txt')
    with open(config_path, 'w') as f:
        f.write("Experiment Configuration\n")
        f.write("=========================\n\n")
        f.write("Models:\n")
        f.write("1. Neuron-level Popup Model: Prunes entire neurons using a score-based selection method.\n")
        f.write("2. Weight-level Popup Model: Prunes individual weights using a score-based selection method.\n")
        f.write("3. Standard MLP: A fully connected network trained normally.\n\n")
        f.write("Hyperparameters:\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Fraction of neurons/weights to keep (k): {k}\n\n")
        f.write("Dataset:\n")
        f.write("Fashion MNIST (28x28 grayscale images, 10 classes)\n\n")
        f.write("Architecture:\n")
        f.write(f"Input dimension: {input_dim}\n")
        f.write(f"Hidden dimension: {hidden_dim}\n")
        f.write(f"Output dimension: {output_dim}\n\n")
        f.write(f"Results directory: {results_dir}\n")

    logger.info(f"Configuration saved to {config_path}")

    # Initialize models
    if TRAIN_NEURON_MODEL:
        neuron_model = NeuronPopupNetwork(input_dim, hidden_dim, output_dim, k=k).to(device)
        optimizer_neuron = optim.SGD(neuron_model.parameters(), lr=learning_rate, momentum=0.9)
        neuron_train_acc = []
        neuron_test_acc = []

    if TRAIN_WEIGHT_MODEL:
        weight_model = WeightPopupNetwork(input_dim, hidden_dim, output_dim, k=k).to(device)
        optimizer_weight = optim.SGD(weight_model.parameters(), lr=learning_rate, momentum=0.9)
        weight_train_acc = []
        weight_test_acc = []

    if TRAIN_STANDARD_MODEL:
        standard_model = StandardMLP(input_dim, hidden_dim, output_dim).to(device)
        optimizer_standard = optim.SGD(standard_model.parameters(), lr=learning_rate, momentum=0.9)
        standard_train_acc = []
        standard_test_acc = []
    
    # Training and evaluation loop
    if TRAIN_NEURON_MODEL:
        logger.info("Training Neuron-level Popup Model")
        logger.info(f"Neuron Model - Initial Train Accuracy: {test(neuron_model, device, train_loader):.2f}%")
        for epoch in range(1, epochs + 1):
            _, acc = train(neuron_model, device, train_loader, optimizer_neuron, epoch)
            test_acc = test(neuron_model, device, test_loader)
            neuron_train_acc.append(acc)
            neuron_test_acc.append(test_acc)
            logger.info(f"Neuron Model - Epoch {epoch}: Test Accuracy: {test_acc:.2f}%")
    
    if TRAIN_WEIGHT_MODEL:
        logger.info("\nTraining Weight-level Popup Model (Original Edge-Popup)")
        logger.info(f"Weight Model - Initial Train Accuracy: {test(weight_model, device, train_loader):.2f}%")
        for epoch in range(1, epochs + 1):
            _, acc = train(weight_model, device, train_loader, optimizer_weight, epoch)
            test_acc = test(weight_model, device, test_loader)
            weight_train_acc.append(acc)
            weight_test_acc.append(test_acc)
            logger.info(f"Weight Model - Epoch {epoch}: Test Accuracy: {test_acc:.2f}%")
    
    if TRAIN_STANDARD_MODEL:
        logger.info("\nTraining Standard MLP Model")
        logger.info(f"Standard Model - Initial Train Accuracy: {test(standard_model, device, train_loader):.2f}%")
        for epoch in range(1, epochs + 1):
            _, acc = train(standard_model, device, train_loader, optimizer_standard, epoch)
            test_acc = test(standard_model, device, test_loader)
            standard_train_acc.append(acc)
            standard_test_acc.append(test_acc)
            logger.info(f"Standard MLP - Epoch {epoch}: Test Accuracy: {test_acc:.2f}%")

    # ----------------------------
    # Plotting the results
    # ----------------------------
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 5))
    plt.suptitle(f"2 Hidden Layers size: {hidden_dim}", fontsize=16)
    
    if TRAIN_NEURON_MODEL:
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, neuron_train_acc, label='Neuron-level Train Accuracy')
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, neuron_test_acc, label='Neuron-level Test Accuracy')
    
    if TRAIN_WEIGHT_MODEL:
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, weight_train_acc, label='Weight-level Train Accuracy')
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, weight_test_acc, label='Weight-level Test Accuracy')
    
    if TRAIN_STANDARD_MODEL:
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, standard_train_acc, label='Standard MLP Train Accuracy')
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, standard_test_acc, label='Standard MLP Test Accuracy')

    plt.subplot(1, 2, 1)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplot(1, 2, 2)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the plot image in the results directory
    plot_path = os.path.join(results_dir, 'training_results.png')
    plt.savefig(plot_path)
    logger.info(f"Plot saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    # ----------------------------
    # Create directory for saving results
    # ----------------------------
    now = datetime.datetime.now().strftime("%d-%m-%y_%H-%M-%S")
    results_dir = os.path.join("results_2hl", now)
    os.makedirs(results_dir, exist_ok=True)
    
    # Set up logging
    log_path = os.path.join(results_dir, 'log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%H-%M-%S',  # Custom time format: hh-mm-ss
        handlers=[
            logging.FileHandler(log_path),  # Log to file
            logging.StreamHandler()        # Log to console
        ]
    )
    logger = logging.getLogger()
    logger.info("Experiment started")

    # Hyperparameters
    batch_size = 64
    epochs = 10
    k = 0.5  # Fraction of neurons/weights to keep for popup models
    learning_rate = 0.01

    # Dimensions (28x28 images, 10 classes)
    input_dim = 28 * 28
    hidden_dim = 256  # Size of hidden layers
    output_dim = 10

    # Models to train
    TRAIN_NEURON_MODEL = True
    TRAIN_WEIGHT_MODEL = False
    TRAIN_STANDARD_MODEL = False

    main()
