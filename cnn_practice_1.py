import unicodedata
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import string
import random
import glob
import os
from torch.utils.data import Subset

# Path to the text folder in Google Drive
folder_path = '/content/drive/MyDrive/names'

# Retrieve file paths of text files in the folder
file_paths = glob.glob(os.path.join(folder_path, '*.txt'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import string
import glob
import os


class NamesDataset(Dataset):
    def __init__(self, data_path):
        self.categories, self.category_lines, self.n_categories, self.n_letters = self.load_data(data_path)
        self.max_length = max(len(line) for lines in self.category_lines.values() for line in lines)
        self.embedding = nn.Embedding(self.n_letters, embedding_dim=100)

    def __len__(self):
        return len(self.category_lines)

    def __getitem__(self, index):
        if index < 0 or index >= len(self.category_lines):
            raise IndexError("Index out of range")

        category = self.categories[index]
        lines = self.category_lines[category]
        line = random.choice(lines)
        category_tensor = torch.tensor([index], dtype=torch.long)
        line_tensor = self.line_to_tensor(line)
        return category_tensor, line_tensor

    def load_data(self, data_path):
        categories = []
        category_lines = {}
        all_letters = string.ascii_letters + " .,;'"

        def unicode_to_ascii(s):
            return ''.join(
                c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn'
                and c in all_letters
            )

        def read_lines(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.read().strip().split('\n')
                return [unicode_to_ascii(line) for line in lines]

        n_letters = len(all_letters)

        for filename in glob.glob(data_path):
            category = os.path.splitext(os.path.basename(filename))[0]
            categories.append(category)
            lines = read_lines(filename)
            category_lines[category] = lines

        n_categories = len(categories)

        return categories, category_lines, n_categories, n_letters

    def line_to_tensor(self, line):
        tensor = torch.zeros(self.max_length, dtype=torch.long)
        for li, letter in enumerate(line):
            tensor[li] = self.letter_to_index(letter)
        return tensor

    def letter_to_index(self, letter):
        all_letters = string.ascii_letters + " .,;'"
        return all_letters.index(letter)


data_path = 'C:/Users/Admin/OneDrive/Desktop/internship/data/data/names/*.txt'


# Create the dataset
names_dataset = NamesDataset(data_path)



#print(names_dataset.category_lines)

#print(names_dataset.categories)

#print(names_dataset.n_categories)

#print(names_dataset.n_letters)

# Create a DataLoader for the dataset
batch_size = 32
data_loader = DataLoader(names_dataset, batch_size=batch_size, shuffle=True)

# Example usage of the DataLoader
for batch in data_loader:
    inputs, targets = batch
    print("Batch inputs:", inputs)
    print("Batch targets:", targets)
    print("Batch size:", inputs.size(0))
    break

max_length = 0

# Iterate over the lines in the dataset
for lines in names_dataset.category_lines.values():
    for line in lines:
        line_length = len(line)
        if line_length > max_length:
            max_length = line_length

print("Max Length:", max_length)

import torch
import torch.nn as nn
import torch.optim as optim


class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_size):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim=100)
        self.conv1 = nn.Conv1d(in_channels=100, out_channels=256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * (max_length // 2), output_size)

    def forward(self, inputs):
        embedded = self.embedding(inputs)  # [batch_size, max_length, embedding_dim]
        embedded = embedded.permute(0, 2, 1)  # [batch_size, embedding_dim, max_length]
        conv_out = self.conv1(embedded)  # [batch_size, out_channels, max_length]
        conv_out = self.relu(conv_out)
        pooled = self.maxpool(conv_out)  # [batch_size, out_channels, max_length // 2]
        flattened = self.flatten(pooled)  # [batch_size, out_channels * (max_length // 2)]
        logits = self.fc(flattened)  # [batch_size, output_size]
        return logits


input_size = names_dataset.n_letters
output_size = names_dataset.n_categories
embedding_size = 100
hidden_size = 128
# Create an instance of the CNN model
cnn_model = CNN(input_size, hidden_size, output_size, embedding_size)

# Print the model architecture
print(cnn_model)

# Split the dataset into training, validation, and testing sets based on categories
train_datasets = []
valid_datasets = []
test_datasets = []
train_ratio = 0.8
valid_ratio = 0.1

for category, lines in names_dataset.category_lines.items():
    category_indices = list(range(len(lines)))
    train_size = int(train_ratio * len(category_indices))
    valid_size = int(valid_ratio * len(category_indices))
    train_indices = category_indices[:train_size]
    valid_indices = category_indices[train_size:train_size + valid_size]
    test_indices = category_indices[train_size + valid_size:]
    train_data = Subset(names_dataset, train_indices)
    valid_data = Subset(names_dataset, valid_indices)
    test_data = Subset(names_dataset, test_indices)
    train_datasets.append(train_data)
    valid_datasets.append(valid_data)
    test_datasets.append(test_data)

train_dataset = torch.utils.data.ConcatDataset(train_datasets)
valid_dataset = torch.utils.data.ConcatDataset(valid_datasets)
test_dataset = torch.utils.data.ConcatDataset(test_datasets)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(valid_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

cnn_model = CNN(input_size, hidden_size, output_size, embedding_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)

num_epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model.to(device)

cnn_model.train()

for epoch in range(num_epochs):
    total_loss = 0.0

    for batch_inputs, batch_targets in train_loader:
        # Move the batch tensors to the appropriate device
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        optimizer.zero_grad()
        logits = cnn_model(batch_inputs)
        loss = criterion(logits, batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss}")

    # Validation
    cnn_model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0

        # Iterate over the validation data batches
        for val_inputs, val_targets in valid_loader:
            val_inputs = val_inputs.to(device)
            val_targets = val_targets.to(device)

            # Forward pass
            val_logits = cnn_model(val_inputs)

            # Calculate the predicted labels
            val_predictions = torch.max(val_logits, 1)

            # Update the total number of correct predictions and samples
            total_correct += (val_predictions == val_targets).sum().item()
            total_samples += val_targets.size(0)

        # Calculate the validation accuracy
        accuracy = total_correct / total_samples
        print(f"Validation Accuracy: {accuracy}")

    # Set the model back to training mode
    cnn_model.train()
