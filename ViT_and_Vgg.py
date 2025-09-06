"""Important imports"""

import torch
import torchvision
from pathlib import Path
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim



"""How to unzip a zip file that contains the training data."""

import os
import requests
import zipfile
from pathlib import Path

# Setup path to data folder
data_path = Path("data/")
image_path = data_path / "face"

# If the image folder doesn't exist, download it and prepare it...
if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)


# Unzip
with zipfile.ZipFile(data_path / "face.zip", "r") as zip_ref:
    print("Unzipping data...")
    zip_ref.extractall(image_path)

# Remove zip file
os.remove(data_path / "face.zip")



"""How to transform the datapoints(images), create datasets and dataloaders."""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

batch_size = 64
train_dir = Path("data/face/train/")
test_dir = Path("data/face/test/")
IMG_SIZE = 224


# Create transform pipeline manually

transform = transforms.Compose([
    transforms.Resize((224, 224)),      # Resize to 224x224
    transforms.ToTensor(),              # Convert to tensor [0,1]
    transforms.RandomRotation(9),
    transforms.Normalize(               # Normalize using ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Use ImageFolder to create dataset(s)
train_data = datasets.ImageFolder(train_dir, transform=transform)
test_data = datasets.ImageFolder(test_dir, transform=transform)

# Get class names
class_names = train_data.classes

# Turn images into data loaders
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
)
test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False, # don't need to shuffle test data
    pin_memory=True,)





"""How to load a pretrained ViT (Vision Transformer model) from torchvision.models, freeze all the parameters so that the training doesn't affect them, and replace the classification head so that it output matches the number of classes of that specific problem."""

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load the pretrained ViT model
model = models.vit_b_16(pretrained=True)

# Freeze the feature extractor
for param in model.parameters():
    param.requires_grad = False

# Replace the classification head for 7 classes
model.heads = nn.Sequential(
    nn.Linear(model.hidden_dim, 7)
)

model.to(device)

# Loss function and optimizer (only train classifier head)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.heads.parameters(), lr=1e-3)

"""The training loop"""

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    accuracy = correct / total * 100
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")




"""How to put the model into evaluation mode, use torch.no_grad() and calculate the validation/test accuracy by using data from from the test dataloader."""

# (Optional) Validation loop
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

val_accuracy = correct / total * 100
print(f"Validation Accuracy: {val_accuracy:.2f}%")




"""How to load a pretrained VGG model from torchvision.models, freeze all the parameters so that the training doesn't affect them, and replace the classification head so that it output matches the number of classes of that specific problem."""

model = models.vgg11(pretrained=True)

# Freeze all feature extractor layers
for param in model.features.parameters():
    param.requires_grad = False

# Replace the classifier for 7 classes
model.classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 7)  # 7 classes
)
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

# Loss and optimizer (train only classifier)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)




"""The training loop"""

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    accuracy = correct / total * 100
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")





"""How to save the weights of a model via a .pth (.pt) file"""

torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': epoch_loss,
}, 'vgg11_checkpoint.pth')

"""How to load the weights of a model which has been saved in a .pth file."""

# Load model and optimizer
checkpoint = torch.load('vgg11_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Optional: resume from the next epoch
start_epoch = checkpoint['epoch'] + 1






"""How to put the model into evaluation mode, use torch.no_grad() and calculate the validation/test accuracy by using data from from the test dataloader."""

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

val_accuracy = correct / total * 100
print(f"Validation Accuracy: {val_accuracy:.2f}%")

"""How to adjust the learning rate of the optimizer"""

for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.5  # or 0.1





"""How to refactor the code block that calculates the test accuracy of a model into a function whose inputs are the model, the test dataloader and the device on which the model is."""

def evaluate_accuracy(model, test_loader, device):
    model.eval()  # set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # disable gradient computation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

print(evaluate_accuracy(model, test_loader, device))



"""How to install and use the summary tool from torchinfo to get a summary of a models architecture, the input and output shapes of each layer and the number of parameters in each layer."""

import torchinfo
from torchinfo import summary

summary(model=model,
        input_size=(32, 3, 224, 224), # (batch_size, color_channels, height, width)
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)

