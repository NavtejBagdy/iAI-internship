import torch
import pandas as pd
from transformers import TapasTokenizer, TapasForQuestionAnswering

# Load the trained model and tokenizer
model_path = "trained_model"
model = TapasForQuestionAnswering.from_pretrained(model_path)
tokenizer = TapasTokenizer.from_pretrained(model_path)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

data = {'Actors': ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
        'Age': ["56", "45", "59"],
        'Number of movies': ["87", "53", "69"],
        'Date of birth': ["7 february 1967", "10 june 1996", "28 november 1967"]}
queries = ["How many movies has George Clooney played in?", "How old is he?", "What's his date of birth?"]
table = pd.DataFrame(data)
# Tokenize the inputs
inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")
inputs = {key: val.to(device) for key, val in inputs.items()}

# Perform the inference
with torch.no_grad():
    outputs = model(**inputs)

# Convert the logits to predicted answer coordinates
predicted_answer_coordinates, = tokenizer.convert_logits_to_predictions(inputs, outputs.logits.cpu().detach())

# Retrieve the answers from the table
answers = []
for coordinates in predicted_answer_coordinates:
    if len(coordinates) == 1:
        answers.append(table.iat[coordinates[0]])
    else:
        cell_values = []
        for coordinate in coordinates:
            cell_values.append(table.iat[coordinate[0], coordinate[1]])
        answers.append(", ".join(cell_values))

# Display the table
print("\nTable:")
print(table.to_string(index=False))

# Display the predicted answers
print("\nPredicted Answers:")
for query, answer in zip(queries, answers):
    print(query)
    print("Predicted answer:", answer)
    print()
