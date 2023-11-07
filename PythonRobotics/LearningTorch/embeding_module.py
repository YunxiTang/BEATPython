import torch
import torch.nn as nn

if __name__ == '__main__':
    n, d, m = 3, 5, 7
    embeding = nn.Embedding(n, d)
    idx = torch.LongTensor([1, 2])
    print(embeding(idx))