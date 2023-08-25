import torch
import pandas as pd
from transformers import TapasTokenizer, TapasForQuestionAnswering, TapasConfig, AdamW

from transformers import TapasConfig, TapasForQuestionAnswering

# for example, the base sized model with default SQA configuration
model = TapasForQuestionAnswering.from_pretrained("google/tapas-base")

# or, the base sized model with WTQ configuration
config = TapasConfig.from_pretrained("google/tapas-base-finetuned-wtq")
model = TapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)

# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available")
    device = torch.device("cuda")
else:
    print("GPU is not available")
    device = torch.device("cpu")
model_name = "google/tapas-base"
tokenizer = TapasTokenizer.from_pretrained(model_name)
# Define the dataset
data = {"sno":["1","2"], "birthdate":["02/03/96","09/03/67"],"name": ["jack", "Ruby",], "job": ["carpenter", "professor"]}
queries = [
   "What is the sno of jack?",
    "What is the sno of Ruby?",
     "What is the birthdate of jack?",
    "What is the birth of Ruby?",
    "What is the job of jack?",
    "What is the job of Ruby?",
    "Give the name with the carpenter job?",
    "Give the name with the professor job?",

]
answer_coordinates = [[(0, 0)],[(1, 0)],[(0, 1)],[(1, 1)],[(0, 2)],[(1, 2)],[(0, 3)],[(1, 3)]]
answer_text = [["1"], ["2"], ["02/03/96"],["09/03/67"], ["jack"], ["Ruby"], ["carpenter"], ["professor"]]
table = pd.DataFrame.from_dict(data)

# Define the model and tokenizer
model_name = "google/tapas-base"
tokenizer = TapasTokenizer.from_pretrained(model_name)
model = TapasForQuestionAnswering.from_pretrained(model_name, config=TapasConfig.from_pretrained(model_name))
model = model.to(device)

# Tokenize the inputs
inputs = tokenizer(
    table=table,
    queries=queries,
    answer_coordinates=answer_coordinates,
    answer_text=answer_text,
    padding="max_length",
    return_tensors="pt",
)

# Define the dataset and dataloader
class TableDataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.inputs.items()}

    def __len__(self):
        return len(self.inputs["input_ids"])

dataset = TableDataset(inputs)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
for epoch in range(2):
    total_loss = 0  # Track the total loss for the epoch
    for step, batch in enumerate(dataloader):
        # Get the inputs
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["labels"]

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

        # Compute the loss and perform backward pass
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()

        # Update the weights
        optimizer.step()

        # Print the loss every 10 steps
        if (step + 1) % 10 == 0:
            average_loss = total_loss / (step + 1)
            print(f"Epoch [{epoch+1}/{20}], Step [{step+1}/{len(dataloader)}], Loss: {average_loss:.4f}")

    # Print the average loss for the epoch
    average_epoch_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{20}], Average Loss: {average_epoch_loss:.4f}")

# Define the path to save the trained model
save_path = "trained_model"

# Save the model
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("Trained model saved at:", save_path)

model = TapasForQuestionAnswering.from_pretrained("trained_model")
model.eval()

