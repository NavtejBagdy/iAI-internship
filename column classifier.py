import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

dataset = [
    ('ID', 'sno'),
    ('Reference Number', 'sno'),
    ('Index', 'sno'),
    ('Code', 'sno'),
    ('Identifier', 'sno'),
    ('Catalog Number', 'sno'),
    ('Sequence', 'sno'),
    ('Order', 'sno'),
    ('Ticket Number', 'sno'),
    ('Barcode', 'sno'),
    ('Tracking Number', 'sno'),
    ('Item Number', 'sno'),
    ('Invoice Number', 'sno'),
    ('Registration Number', 'sno'),
    ('Membership Number', 'sno'),
    ('Unique ID', 'sno'),
    ('Lot Number', 'sno'),
    ('Reference Code', 'sno'),
    ('Model Number', 'sno'),
    ('Case Number', 'sno'),
    ('Part Number', 'sno'),
    ('Serial Code', 'sno'),
    ('Identifier Key', 'sno'),
    ('Record ID', 'sno'),
    ('Ticket Code', 'sno'),

    ('Title', 'names'),
    ('Full Name', 'names'),
    ('Last Name', 'names'),
    ('Given Name', 'names'),
    ('Alias', 'names'),
    ('Nickname', 'names'),
    ('Middle Name', 'names'),
    ('Maiden Name', 'names'),
    ('Family Name', 'names'),
    ('Surname', 'names'),
    ('Preferred Name', 'names'),
    ('Initials', 'names'),
    ('First Name', 'names'),
    ('Pen Name', 'names'),
    ('Stage Name', 'names'),
    ('Screen Name', 'names'),
    ('Username', 'names'),
    ('Display Name', 'names'),
    ('Caller Name', 'names'),
    ('Contact Name', 'names'),
    ('Business Name', 'names'),
    ('Client Name', 'names'),
    ('Guest Name', 'names'),
    ('Subscriber Name', 'names'),
    ('Author Name', 'names'),


    ('Event Date', 'date'),
    ('Transaction Date', 'date'),
    ('Effective Date', 'date'),
    ('Due Date', 'date'),
    ('Creation Date', 'date'),
    ('Start Date', 'date'),
    ('End Date', 'date'),
    ('Published Date', 'date'),
    ('Modification Date', 'date'),
    ('Expiry Date', 'date'),
    ('Release Date', 'date'),
    ('Closing Date', 'date'),
    ('Birthdate', 'date'),
    ('Anniversary Date', 'date'),
    ('Renewal Date', 'date'),
    ('Completion Date', 'date'),
    ('Last Updated', 'date'),
    ('Scheduled Date', 'date'),
    ('Event Time', 'date'),
    ('Booking Date', 'date'),
    ('Invoice Date', 'date'),
    ('Due Time', 'date'),
    ('Enrollment Date', 'date'),
    ('Valid From', 'date'),
    ('Maturity Date', 'date'),

    ('Position', 'job'),
    ('Role', 'job'),
    ('Title', 'job'),
    ('Occupation', 'job'),
    ('Profession', 'job'),
    ('Specialization', 'job'),
    ('Field', 'job'),
    ('Career', 'job'),
    ('Industry', 'job'),
    ('Job Category', 'job'),
    ('Employment Type', 'job'),
    ('Job Level', 'job'),
    ('Division', 'job'),
    ('Department', 'job'),
    ('Team', 'job'),
    ('Function', 'job'),
    ('Responsibility', 'job'),
    ('Work Area', 'job'),
    ('Work Role', 'job'),
    ('Vocation', 'job'),
    ('Calling', 'job'),
    ('Craft', 'job'),
    ('Trade', 'job'),
    ('Position Title', 'job'),
    ('Proficiency', 'job'),
]
input_vocab = set()
output_vocab = set()

for input_text, output_text in dataset:
    input_vocab.update(input_text.split())
    output_vocab.add(output_text)

input_vocab = set()
output_vocab = set()

for input_text, output_text in dataset:
    input_vocab.update(input_text)
    output_vocab.add(output_text)

input_vocab = sorted(input_vocab)
output_vocab = sorted(output_vocab)
input_to_index = {token: i for i, token in enumerate(input_vocab)}
output_to_index = {token: i for i, token in enumerate(output_vocab)}


numerical_dataset = []
for input_text, output_text in dataset:
    input_indices = [input_to_index[char] for char in input_text]
    output_index = output_to_index[output_text]
    numerical_dataset.append((input_indices, output_index))
class ColumnNameClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ColumnNameClassifier, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.conv1 = nn.Conv1d(hidden_size, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)  # Reshape for Conv1d input
        conv_output = self.conv1(embedded)
        conv_output = self.relu(conv_output)
        pooled_output = torch.max(conv_output, dim=2)[0]
        output = self.fc(pooled_output)
        return output

input_size = len(input_vocab)
hidden_size = 16
output_size = len(output_vocab)
num_epochs = 100
batch_size = 4
learning_rate = 0.01
model = ColumnNameClassifier(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    total_loss = 0
    for input_indices, output_index in numerical_dataset:
        optimizer.zero_grad()
        inputs = torch.tensor(input_indices, dtype=torch.long)
        targets = torch.tensor(output_index, dtype=torch.long)

        inputs = inputs.unsqueeze(0)  # Add a dimension for batch size
        targets = targets.unsqueeze(0)  # Add a dimension for batch size

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


        total_loss += loss.item()

    if (epoch+1) % 10 == 0:
        print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}')

def predict_column(input_text):
    input_indices = [input_to_index.get(token, -1) for token in input_text]
    input_indices = [index for index in input_indices if index != -1]
    if len(input_indices) == 0:
        return "Unknown"
    input_tensor = torch.tensor(input_indices, dtype=torch.long)
    output_tensor = model(input_tensor.unsqueeze(0))
    predicted_index = output_tensor.argmax().item()
    predicted_output = output_vocab[predicted_index]
    return predicted_output


print(predict_column('release date'))
print(predict_column('serial code'))
print(predict_column('Alias'))
print(predict_column('profession'))