import torch
import torch.nn as nn


if __name__ == "__main__":
    batch_size = 3
    src_len = 5
    d_model = 32
    nhead = 8
    num_layer = 6

    # input sequence
    src_seq = torch.randn([batch_size, src_len, d_model])

    # encoder mask (batch_size x src_len)
    encoder_mask = torch.tensor(
        [[0, 0, 0, 1, 1], [0, 0, 1, 1, 1], [0, 0, 0, 0, 1]]
    ).bool()

    print(src_seq, "\n", encoder_mask)

    encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
    encoder = nn.TransformerEncoder(encoder_layer, num_layer)
    output = encoder(src_seq, src_key_padding_mask=encoder_mask)
    print(output.shape)
