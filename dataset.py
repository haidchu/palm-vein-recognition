from torch.utils.data import Dataset

class DorsalDataset(Dataset):
    def __init__(self, data, labels, transform=None) -> None:
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = self.data[index]

        if self.transform:
            image = self.transform(image)

        label = self.labels[index]

        return image, label
