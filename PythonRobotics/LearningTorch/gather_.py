import torch

if __name__ == '__main__':
    tensor_0 = torch.randint(0, 5, (2, 5, 3))
    print(tensor_0)
    index = torch.tensor([[2, 1, 0],
                          [1, 1, 2]])
    print( '=================' )
    indices = index.unsqueeze(-1).repeat(1, 1, tensor_0.shape[-1])
    print( indices.shape )
    print( '=================' )
    print( indices, indices.shape )
    tensor_1 = torch.gather(tensor_0, 1, indices)
    print( '=================' )
    print(tensor_1)