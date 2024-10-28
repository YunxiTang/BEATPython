import torch
from torch import nn, Tensor


class EncoderLayer(nn.Module):
    def __init__(self, dim_model, nhead, dim_feedforward, dropout, pre_norm: bool=False):
        super(EncoderLayer, self).__init__()
        self.pre_norm = pre_norm
        self.self_attn = nn.MultiheadAttention(embed_dim=dim_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.linear2 = nn.Linear()

    def forward(self, x, pos_embed: Tensor = None, key_padding_mask: Tensor = None):
        skip = x
        if self.pre_norm:
            x = self.norm1(x)

        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]
        x = skip + self.dropout1(x)

        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x

        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x

if __name__ == '__main__':
    model = EncoderLayer(32, 4, 64, 0.2)
    x = torch.randn([2, 10, 32])
    y = model(x)
    print(y.shape)