from datasets import MNIST28
from torchvision.transforms import ToPILImage, Compose, ToTensor, Lambda
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.unet import UNet

import torch

def test_unet():

    dataset = MNIST28(
        transform=Compose([
            ToTensor(),
            Lambda(lambda x: x*2 - 1)
        ])
    )

    loader = DataLoader(dataset, batch_size=32)

    unet = UNet()

    x_test = next(iter(loader))

    checkpoint_location = "checkpoints/unet-test.pt"
    # train the unet
    unet.train()
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(unet.parameters(), lr=3e-4)

    for epoch in range(100):
        y = unet(x_test)
        loss = loss_fn(y, x_test)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"epoch {epoch} -- loss: {loss.item()}")
        torch.save(unet.state_dict(), checkpoint_location)

    # evaluate the model
    unet.load_state_dict(torch.load(checkpoint_location))
    unet.eval()
    with torch.no_grad():
        y_test = unet(x_test)

    y_test = torch.cat([y_test, x_test], dim=0)
    for i, y in enumerate(y_test):
        y_test[i] = (y - y.min())/(y.max() - y.min())

    print(y_test.shape)
    grid = make_grid(y_test)#*0.5 + 0.5)
    grid = ToPILImage()(grid)
    plt.imshow(grid)
    plt.show()