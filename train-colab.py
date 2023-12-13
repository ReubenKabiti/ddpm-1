from pipelines import SimplePipeline
from torchvision.transforms import Compose, ToTensor, Lambda, Resize, CenterCrop
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num-examples", "-n", required=True)
parser.add_argument("--num-epochs", "-e", required=True)
args = parser.parse_args()

num_examples = int(args.num_examples)
num_epochs = int(args.num_epochs)

dataset = MNIST(
    root="datasets",
    download=True,
    transform=Compose([
        ToTensor(),
        Lambda(lambda x: x*2 - 1),
        Resize((28, 28)),
        CenterCrop((28, 28))
    ])
)

dataset = Subset(dataset, list(range(num_examples)))
loader = DataLoader(dataset, batch_size=32)


pipeline = SimplePipeline()

checkpoint_dir = "checkpoints/pipeline-test.pt"

pipeline.train(loader, checkpoint_dir=checkpoint_dir, epochs=num_epochs)
