import torch
from einops import rearrange, repeat
import torch.nn as nn


def test1():
    cls_token = nn.Parameter(torch.rand(1, 1, 7), requires_grad=True)
    repeated_cls = cls_token.repeat(2, 1, 1)
    repeated_cls_v2 = repeat(cls_token, 'a b c -> (repeat a) b c', repeat=2)
    print(cls_token)
    print(repeated_cls - repeated_cls_v2)
    print(repeated_cls.shape)
    print(repeated_cls_v2.shape)


    input = torch.tensor([[0.0, 0.1, 0.2, 0.3],
                          [1.0, 1.1, 1.2, 1.3],
                          [2.0, 2.1, 2.2, 2.3]])
    length = torch.LongTensor([[2, 2, 2, 2],
                               [1, 1, 1, 1],
                               [0, 0, 0, 0],
                               [0, 1, 2, 0]])
    out = torch.gather(input, dim=0, index=length)
    print(out)


    print( '**********==================================**********' )
    tensor_0 = torch.randint(0, 5, (2, 5, 3))
    print( tensor_0 )
    index = torch.tensor([[2, 1, 0],
                          [1, 1, 2]]).long()
    
    print( '=================' )
    indices = index.unsqueeze(-1)
    print( 'modified indice shape: \n', indices.shape )
    print( 'modified indice      : \n', indices )

    indices = indices.repeat(1, 1, tensor_0.shape[-1])
    print( 'modified indice shape: \n', indices.shape )
    print( 'modified indice      : \n', indices )

    tensor_1 = torch.gather(tensor_0, 1, indices)
    print( '=================' )
    print(tensor_0)
    print(tensor_1)


def test2():
    latent_z = torch.randint(1, 5, (2, 5, 3))
    print( latent_z )
    mask_idx = torch.tensor([[0,1],
                             [0,2]])
    indices = mask_idx.unsqueeze(-1).repeat(1, 1, latent_z.shape[-1])
    tmp = torch.gather(latent_z, dim=1, index=indices)
    print( '=============================' )
    print( tmp )
    return tmp

def test3():
    tokens = torch.randn(3, 5, 4)
    # Index tensor (batch_size x num_indices), containing the indices to gather
    selected_indices = torch.tensor([[0, 2], 
                                     [1, 3], 
                                     [0, 4]])
    # We need to unsqueeze the indices to match the dimensions of `tokens`
    # indices shape should match tokens in all dimensions except dim=1
    # (batch_size x num_indices x embedding_dim)
    indices = selected_indices.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
    
    # Use torch.gather to select the tokens at the specified indices
    selected_tokens = torch.gather(tokens, dim=1, index=indices)

    print("Original Tokens Tensor:")
    print(tokens)
    print("\nSelected Tokens Tensor:")
    print(selected_tokens)
    
def test4():
    # Example input tensor (batch_size x seq_len x embedding_dim)
    tokens = torch.randn(3, 5, 4)  # 3 sequences, each of length 5, embedding dimension of 4

    # Index tensor (batch_size x num_indices), containing the indices to gather
    indices = torch.tensor([[0, 2], [1, 3], [0, 4]])  # Select 2 tokens for each sequence

    # Use advanced indexing to directly select the tokens
    batch_indices = torch.arange(tokens.size(0)).unsqueeze(1)  # Create batch indices (batch_size x 1)

    # Advanced indexing to select specific tokens
    selected_tokens = tokens[batch_indices, indices]

    print("Original Tokens Tensor:")
    print(tokens)
    print("\nSelected Tokens Tensor:")
    print(selected_tokens)


if __name__ == '__main__':
    test4()
    