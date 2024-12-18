import torch
import torch.nn as nn


if __name__ == '__main__':
    model = nn.Embedding(20, 5)
    pos = torch.arange(0, 3, dtype=torch.long)
    print(pos, pos.shape, pos.dtype)
    pos_emb = model(pos)
    print(pos_emb, pos_emb.shape)