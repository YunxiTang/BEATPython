from transformers import BertTokenizer, BertModel
import torch

model_name = 'bert-base-uncased'

# Load pretrained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Example material description
description1 = "high stiffness, low damping, low friction"
description2 = "middle stiffness, low damping"
description3 = "low stiffness, high damping, zero friction"
description = [description1, description2, description3]
# Tokenize and encode the description
inputs = tokenizer(description, return_tensors='pt', truncation=True, padding=True)

# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state # last hidden state
    print(last_hidden_state.shape)
    embedding = last_hidden_state[:,0,:]
    print(embedding.shape)

cos_sim = torch.nn.CosineSimilarity(dim=0)
print(cos_sim(embedding[0], embedding[1]))
print(cos_sim(embedding[1], embedding[2]))
print(embedding[0], embedding[0] - embedding[2])
