from datasets import MNIST28
from torchvision.transforms import ToPILImage, Compose, ToTensor, Lambda
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.unet import UNet
from schedulars.linear import LinearSchedular
from trainer import Trainer
from tqdm import tqdm
from copy import deepcopy
from generator import generate, to_pil

import torch

def load_data():
    dataset = MNIST28(
        transform=Compose([
            ToTensor(),
            Lambda(lambda x: x*2 - 1)
        ])
    )

    loader = DataLoader(dataset, batch_size=32)
    return loader


def test_unet():

    loader = load_data()

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


def test_schedular():
    loader = load_data()
    schedular = LinearSchedular(400, 0.0001, 0.002)
    x = next(iter(loader))[:1]
    y = deepcopy(x)

    for t in tqdm(torch.arange(1, schedular.num_steps, 10)):
        noised, _ = schedular.add_noise(y[-1:], t)
        y = torch.cat([y, noised], dim=0)
    
    for i, item in enumerate(y):
        y[i] = (item - item.min())/(item.max() - item.min())
    print(y.shape, x.shape)
    # y = y*0.5 + 0.5
    grid = make_grid(y)
    grid = ToPILImage()(grid)
    plt.imshow(grid)
    plt.show()


def test_trainer():
    loader = load_data()
    unet = UNet()
    schedular = LinearSchedular(400, 0.0001, 0.002)
    trainer = Trainer(unet, schedular, "checkpoints/trainer-test.pt")
    trainer.train(loader)

def test_generator():
    schedular = LinearSchedular(400, 0.0001, 0.002)
    img = generate(schedular, model_path="checkpoints/trainer-test.pt")
    img = to_pil(img)
    plt.imshow(img)


test_generator()