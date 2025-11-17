import torch


def compute_acc_score(x, y):
    batch_size = x.shape[0]
    correct_num = 0
    for idx in range(batch_size):
        if all(x[idx] == y[idx]):
            correct_num += 1
    return correct_num


def compute_acc_score_v2(x, y):
    num_feat = x.shape[1]
    correct_num = 0
    print(x == y)
    tmp = torch.sum(x == y, dim=1)
    tmp2 = tmp == (torch.ones_like(tmp)) * num_feat
    correct_num = torch.sum(tmp2)
    return correct_num


if __name__ == "__main__":
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [3.6, 8.9, 3.2]])
    y = torch.tensor([[1, 2, 3], [4, 5, 6], [3.6, 8.9, 3.1]])
    # compute_acc_score_v2(x, y)
    print(compute_acc_score_v2(x, y))
