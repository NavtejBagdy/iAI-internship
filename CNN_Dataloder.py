import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class IMDBDataset(Dataset):
    def __init__(self, file_path, max_seq_len):
        self.df = pd.read_csv(file_path)
        self.max_seq_len = max_seq_len
        self.vocab, self.char_to_index = self.create_vocab_and_mapping()
        self.sequences, self.labels = self.preprocess_data()

    def create_vocab_and_mapping(self):
        vocab = set()
        for text in self.df['review']:
            vocab.update(text)
        char_to_index = {char: i + 1 for i, char in enumerate(vocab)}
        char_to_index['<PAD>'] = 0
        return vocab, char_to_index

    def preprocess_data(self):
        positive_df = self.df[self.df['sentiment'] == 'positive'].head(150)
        negative_df = self.df[self.df['sentiment'] == 'negative'].head(150)
        df_reduced = pd.concat([positive_df, negative_df])
        df_reduced['review'] = df_reduced['review'].apply(lambda x: ' '.join(x.split()[:50]))
        df_reduced['review'] = df_reduced['review'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s\W]', '', x))
        df_reduced.reset_index(drop=True, inplace=True)

        sequences = []
        labels = []
        for text, label in zip(df_reduced['review'], df_reduced['sentiment']):
            seq = [self.char_to_index[char] for char in text]
            if len(seq) > 0:
                sequences.append(torch.tensor(seq))
                labels.append(1 if label == 'positive' else 0)

        return sequences, labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index]
        label = self.labels[index]
        return seq, label

# Load the dataset
file_path = "C:\\Users\\Admin\\OneDrive\\Desktop\\internship\\IMDB Dataset.csv\\IMDB Dataset.csv"

# Create the dataset
max_seq_len = 50
dataset = IMDBDataset(file_path, max_seq_len)

# Split the dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Pad the sequences to a fixed length
train_dataset = [(seq, label) for seq, label in train_dataset]
test_dataset = [(seq, label) for seq, label in test_dataset]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    return sequences, torch.tensor(labels)

# Create data loaders
batch_size = 60  # Adjust batch size as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_filters, filter_sizes, output_size):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_size, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)
        conved = [nn.functional.relu(conv(embedded)) for conv in self.convs]
        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

# Set the hyperparameters
vocab_size = len(dataset.vocab) + 1
embed_size = 100
num_filters = 100
filter_sizes = [3, 4, 5]
output_size = 1
learning_rate = 0.001
num_epochs = 10

model = CNN(vocab_size, embed_size, num_filters, filter_sizes, output_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

# Train the model
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

test_predictions = []
test_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predictions = torch.round(torch.sigmoid(outputs))
        test_predictions.extend(predictions.tolist())
        test_labels.extend(labels.tolist())

from sklearn.metrics import classification_report
report = classification_report(test_labels, test_predictions)
print(report)
