from transformers import BertConfig, BertModel

configuration = BertConfig()

model = BertModel(configuration)
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of Model Param: {total_params / 1e6} M")
# print(model)
