import torch
from torch.utils.data import DataLoader, Dataset



class DummyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = torch.range(0, 99).view(-1, 1)  # 100 samples, 1 feature
        self.labels = torch.randint(0, 2, (100,))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'data': self.data[idx],
            'label': self.labels[idx]
        }
    


dataset = DummyDataset()
dataloader = DataLoader(dataset, batch_size=3, shuffle=False)
for batch in dataloader:
    print(batch)
    break
    # print(batch['data'].shape, batch['label'].shape)
    # print(batch['data'])
    # print(batch['label'])
    # print(batch['data'].dtype)    
