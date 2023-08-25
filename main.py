import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
df=pd.read_csv("C:\\Users\\Admin\\OneDrive\\Desktop\\internship\\IMDB Dataset.csv\\IMDB Dataset.csv")
#print(df.head())

positive_df = df[df['sentiment'] == 'positive'].head(150)
negative_df = df[df['sentiment'] == 'negative'].head(150)
df_reduced = pd.concat([positive_df, negative_df])
df_reduced['review'] = df_reduced['review'].apply(lambda x: ' '.join(x.split()[:50]))
df_reduced['review'] = df_reduced['review'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s\W]', '', x))
df_reduced.reset_index(drop=True, inplace=True)
df_reduced.to_csv('imdb_reduced.csv', index=False)
#df_reduced = pd.read_csv("C:\\Users\\Admin\\PycharmProjects\\Sentence Embedding\\imdb_reduced.csv")

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
vocab = set()
max_seq_len = 0
for text in df_reduced['review']:
    vocab.update(text)
    max_seq_len = max(max_seq_len, len(text))
char_to_index = {char: i+1 for i, char in enumerate(vocab)}
char_to_index['<PAD>'] = 0
sequences = []
for text in df_reduced['review']:
    seq = [char_to_index[char] for char in text]
    seq += [char_to_index['<PAD>']] * (max_seq_len - len(seq))
    sequences.append(seq)

from sklearn.preprocessing import LabelEncoder

# Create label encoder
label_encoder = LabelEncoder()

# Encode the sentiment labels
encoded_labels = label_encoder.fit_transform(df_reduced['sentiment'])

# Convert the sequences to a PyTorch tensor
data = torch.tensor(sequences, dtype=torch.long)
labels = torch.tensor(encoded_labels, dtype=torch.float)

from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)


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
vocab_size = len(vocab) + 1
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
    for i, inputs in enumerate(train_data):
        inputs = inputs.unsqueeze(0)
        targets = train_labels[i].unsqueeze(0).unsqueeze(0)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_data)}')
test_predictions = []
with torch.no_grad():
    for inputs in test_data:
        outputs = model(inputs.unsqueeze(0))
        predictions = torch.round(torch.sigmoid(outputs))
        test_predictions.extend(predictions.tolist())

from sklearn.metrics import classification_report
report = classification_report(test_labels, test_predictions)
print(report)