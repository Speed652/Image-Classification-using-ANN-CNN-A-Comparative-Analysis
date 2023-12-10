import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 dataset loading and preprocessing
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Artificial Neural Network (ANN) model
class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ANN, self).__init__()
        # Define your layers here
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Flatten the input tensor before passing it to the first fully connected layer
        x = x.view(-1, 32*32*3)
        # Define the forward pass
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Convolutional Neural Network (CNN) model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Define your convolutional layers and fully connected layers here
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 16 * 16, 10)

    def forward(self, x):
        # Define the forward pass
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 16 * 16 * 16)  # Reshape for fully connected layer
        x = self.fc1(x)
        return x

# Instantiate models
ann_model = ANN(input_size=32*32*3, hidden_size=128, output_size=10).to(device)
cnn_model = CNN().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
ann_optimizer = optim.SGD(ann_model.parameters(), lr=0.01)
cnn_optimizer = optim.SGD(cnn_model.parameters(), lr=0.01)

# Training loop for ANN
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs, labels = inputs.view(-1, 32*32*3).to(device), labels.to(device)

        # Forward pass
        ann_outputs = ann_model(inputs)
        loss = criterion(ann_outputs, labels)

        # Backward pass and optimization
        ann_optimizer.zero_grad()
        loss.backward()
        ann_optimizer.step()

# Training loop for CNN
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        cnn_outputs = cnn_model(inputs)
        loss = criterion(cnn_outputs, labels)

        # Backward pass & optimization
        cnn_optimizer.zero_grad()
        loss.backward()
        cnn_optimizer.step()

# Evaluation on test set
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

ann_test_accuracy = evaluate(ann_model, test_loader)
cnn_test_accuracy = evaluate(cnn_model, test_loader)

# Printing results
print(f'ANN Test Accuracy: {ann_test_accuracy * 100:.2f}%')
print(f'CNN Test Accuracy: {cnn_test_accuracy * 100:.2f}%')