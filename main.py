from pipelines import SimplePipeline
from datasets import MNIST28
from torchvision.transforms import Compose, ToTensor, Lambda
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

dataset = MNIST28(
    transform=Compose([
        ToTensor(),
        Lambda(lambda x: x*2 - 1)
    ])
)

dataset = Subset(dataset, list(range(100)))
loader = DataLoader(dataset, batch_size=8)


pipeline = SimplePipeline()

checkpoint_dir = "checkpoints/pipeline-test.pt"

# pipeline.train(loader, checkpoint_dir=checkpoint_dir, epochs=10)
pipeline.load_from_file(checkpoint_dir)

imgs = pipeline.generate()
plt.imshow(imgs)
plt.show()