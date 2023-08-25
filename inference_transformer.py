import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
    #python convert_rwkv_checkpoint_to_hf.py --repo_id shahkalp09/RWKVV --checkpoint_file rwkv-10.pth --output_dir converted_weights_3 --size 169M
# Load the converted model
model_dir = "C:\\Users\\Admin\\PycharmProjects\\character-embbeding\\converted_weights_3"
model = AutoModelForCausalLM.from_pretrained(model_dir)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Input text
input_text = "serial number 17 18 19 <start>"

# Tokenize input text
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

# Generate output
output = model.generate(input_ids, max_length=50)

# Decode the output tokens
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the generated text
print("Column Name:", output_text.split("<start>")[1].split("<stop>")[0].strip())



