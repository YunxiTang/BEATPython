import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

print(model)

image = preprocess(Image.open("./teaser.png")).unsqueeze(0).to(device)
text = clip.tokenize(
    ["It is a diagram", "It should be two robots", "I guess it is a tanghulu"]
).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)
