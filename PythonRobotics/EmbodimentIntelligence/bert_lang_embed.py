from transformers import BertTokenizer, BertModel
import torch

# Load pretrained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Example material description
description = "high stiffness, low damping"

# Tokenize and encode the description
inputs = tokenizer(description, return_tensors='pt')

# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling for sentence embedding

# embeddings now holds the vector representation of the description
