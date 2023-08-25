import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

data= [
    ["name","job"],
    ["John", "engineer"],
    ["Emily", "manager"],
    ["Michael", "analyst"],
    ["Sophia", "programmer"],
    ["Olivia", "consultant"],
    ["Ethan", "technician"],
    ["Ava", "sales"],
    ["Jacob", "finance"],
    ["Mia", "teacher"],
    ["Noah", "coordinator"],
    ["Isabella", "assistant"],
    ["William", "administrator"],
    ["James", "programmer"],
    ["Sophia", "specialist"],
    ["Harper", "supervisor"],
    ["Benjamin", "programmer"],
    ["Grace", "assistant"],
    ["Liam", "technician"],
    ["Emma", "manager"],
    ["Charlotte", "programmer"],
    ["Noah", "engineer"],
    ["Ava", "developer"],
    ["Daniel", "analyst"],
    ["Emily", "specialist"],
    ["Michael", "supervisor"],
    ["Lily", "programmer"],
    ["Ethan", "technician"],
    ["Amelia", "sales"],
    ["William", "finance"],
    ["Harper", "teacher"],
    ["Mia", "coordinator"],
    ["Oliver", "assistant"],
    ["Emily", "administrator"],
    ["Jacob", "programmer"],
    ["Ava", "specialist"],
    ["Sophia", "supervisor"],
    ["Benjamin", "programmer"],
    ["Grace", "assistant"],
    ["Liam", "technician"],
    ["Emma", "manager"],
    ["Charlotte", "programmer"],
    ["Noah", "engineer"],
    ["Ava", "developer"],
    ["Daniel", "analyst"],
    ["Emily", "specialist"],
    ["Michael", "supervisor"],
    ["Lily", "programmer"],
    ["Ethan", "technician"],
    ["Amelia", "sales"],
    ["William", "finance"]
]

import torch
from torch.utils.data import Dataset

# Define a custom dataset class for the table data
class TableDataset(Dataset):
    def __init__(self, data):
        self.column_names = data[0]
        self.table_data = data[1:]

    def __len__(self):
        return len(self.table_data)

    def __getitem__(self, index):
        row = self.table_data[index]
        return self.column_names, row

# Create an instance of the custom dataset
dataset = TableDataset(data)

# Split the dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

# Get the column names
column_names_train = train_set.dataset.column_names



test_set.dataset.column_names

# Print the column names
print("Column Names:")
print(column_names_train)

vocabulary = set()
for row in data:
    for token in row:
        vocabulary.update(token)

# Map characters to indices in the vocabulary
char_to_index = {char: index for index, char in enumerate(vocabulary)}

# Separate headers and table data
headers = data[0]
table_data = data[1:]

# Perform character-level embedding and one-hot encoding using the vocabulary set
headers_encoded = [torch.tensor([char_to_index[char] for char in header], dtype=torch.long) for header in headers]
table_data_encoded = []
for row in table_data:
    row_encoded = [torch.tensor([char_to_index[char] for char in value], dtype=torch.long) for value in row]
    table_data_encoded.append(row_encoded)

from torch.nn.utils.rnn import pad_sequence
fixed_length1 = 10
fixed_length2=5
# Pad or truncate the encoded headers
headers_padded = pad_sequence(headers_encoded, batch_first=True, padding_value=0)
headers_padded = headers_padded[:, :fixed_length1]
headers_padded = torch.nn.functional.pad(headers_padded, (0, 10 - headers_padded.size(1)))

headers_padded

# Pad or truncate the encoded table data
table_data_padded = []
for row_encoded in table_data_encoded:
    row_padded = pad_sequence(row_encoded, batch_first=True, padding_value=0)
    row_padded = row_padded[:, :fixed_length2]
    table_data_padded.append(row_padded)

# Pad or truncate the table data rows to have the same number of columns
max_columns = max([row.size(1) for row in table_data_padded])
table_data_padded = [torch.cat([row, torch.zeros(row.size(0), max_columns - row.size(1))], dim=1) for row in table_data_padded]

# Pad or truncate the table data rows to have the same number of rows
max_rows = max([row.size(0) for row in table_data_padded])
table_data_padded = [torch.cat([row, torch.zeros(max_rows - row.size(0), row.size(1))], dim=0) for row in table_data_padded]

len(table_data_padded)
class ColumnNamePredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ColumnNamePredictionModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
       embedded = self.embedding(x.long())
       embedded = embedded.permute(0, 2, 1)  # Reshape for Conv1d input
       hidden = self.conv(embedded.squeeze(2))
       hidden = hidden.permute(0, 2, 1)  # Reshape for Linear input
       output = self.fc(hidden)
       return output



# Set hyperparameters
input_size = 36
hidden_size = 10
output_size = 36  # Number of classes (equal to the number of characters in the vocabulary)

# Create an instance of the model
model = ColumnNamePredictionModel(input_size, hidden_size, output_size)

# Set loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert the table data and headers to torch tensors
table_data_tensor = torch.stack(table_data_padded)
headers_tensor = headers_padded.unsqueeze(1)  # Add a singleton dimension for batch

# Training loop
num_epochs = 1000
batch_size = 2
total_samples = table_data_tensor.size(0)
total_batches = total_samples // batch_size

for epoch in range(num_epochs):
    for batch_idx in range(total_batches):
        # Prepare the batch
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch_table_data = table_data_tensor[start_idx:end_idx]
        batch_headers = headers_tensor
        batch_table_data = batch_table_data.view(batch_size, -1)

        # Forward pass
        logits = model(batch_table_data)

        # Prepare the target tensor
        target = torch.zeros_like(batch_headers.squeeze())
        for i, header in enumerate(batch_headers.squeeze()):
            for j, index in enumerate(header):
                if index.item() > 0:  # Exclude padding valuesa
                    target[i, j] = index

        # Compute loss
        loss = criterion(logits.permute(0, 2, 1), target.long())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the loss for this epoch
    if(epoch + 1) % 100 == 0:
       print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
index_to_char = {index: char for char, index in char_to_index.items()}

def predict(input_data):
    # Convert input to tensor
    input_tensor = torch.tensor(input_data)

    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Forward pass
        logits = model(input_tensor.unsqueeze(0))
        _, predicted_idx = torch.max(logits, dim=2)
        predicted_column_name = "".join([index_to_char[idx.item()] for idx in predicted_idx.squeeze()])

    # Set the model back to training mode
    model.train()

    return predicted_column_name

input_data = [char_to_index[char] for char in "john"]
predicted_column = predict(input_data)
print(f"Predicted column name: {predicted_column}")

