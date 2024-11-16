from torch.utils.data import DataLoader

def create_dataloader_v1(dataset, batch_size=4, shuffle=True, drop_last=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader