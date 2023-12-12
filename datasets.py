from torch.utils.data import Dataset
from idx2numpy import convert_from_file as load_idx


class MNIST28(Dataset):

    def __init__(self, train: bool = True, transform = None):
        if train:
            self.images = load_idx("datasets/mnist/train-images.idx3-ubyte")
        else:
            self.images = load_idx("datasets/mnist/t10k-images.idx3-ubyte")

        self.transform = transform

    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, idx: int):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)

        return image