
"""
using neural network to calculate wasserstein distance
between two data sets.
"""

import torch
from torch.autograd import Variable
import numpy as np

class WassLoss(torch.nn.Module):
    def __init__(self, LAMBDA = 10):
        super(WassLoss, self).__init__()
        self.beta = LAMBDA
        return

    def forward(self, real, fake, gradient_penalty):
        l = fake.mean() - real.mean()
        loss = l + self.beta*gradient_penalty
        print(loss)
        return loss

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.LeakyReLU(),
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(64, 16),
            torch.nn.LeakyReLU(),
        )
        self.fc3 = torch.nn.Sequential(
            torch.nn.Linear(16, 1),
            torch.nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class Wasserstein():
    def __init__(self,real_data, fake_data, Net, batch_size, LAMBDA, iters):
        self.real_data = real_data
        self.fake_data = fake_data
        self.batch_size = batch_size
        self.LAMBDA = LAMBDA
        self.iters = iters
        self.model = Net()

    # used for calc gradient penalty
    def _genIntrpolates(self):
        alpha = torch.rand(self.batch_size,1)
        alpha = alpha.expand(self.real_data.size())
        interpolates = alpha*self.real_data + ((1-alpha)*self.fake_data)
        interpolates = Variable(interpolates, requires_grad=True)
        return interpolates

    def train(self, lr = 1e-4, betas = (0.5,0.9)):
        model = self.model
        loss_func = WassLoss(self.LAMBDA)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)

        for t in range(self.iters):
            optimizer.zero_grad()
            disc_real= model(self.real_data)
            disc_fake= model(self.fake_data)
            interpolates = self._genIntrpolates()
            disc_interpolates = model(interpolates)
            gradients = torch.autograd.grad(
                outputs=disc_interpolates, inputs=interpolates,
                grad_outputs=torch.ones(disc_interpolates.size()),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            loss = loss_func(disc_real, disc_fake, gradient_penalty)
            loss.backward()
            optimizer.step()

    def distance(self,real_data=None, fake_data=None):
        if not real_data:
            real_data = self.real_data
        if not fake_data:
            fake_data = self.fake_data
        w_distance = self.model(real_data).mean() - self.model(fake_data).mean()
        return w_distance


if __name__ == '__main__':
    LAMBDA = 10
    BATCH_SIZE = 100
    ITERS = 3000

    def gen_real_dataset(batch_size):
        x = []
        for i in np.arange(0, 1, 1/batch_size):
            x.append([i, 0])
        real_data = torch.tensor(x).float()
        return real_data

    def gen_fake_dataset(real_data, dist=1):
        fake_data = real_data.numpy().copy()
        fake_data[:, 1] = fake_data[:, 1] + dist
        fake_data = torch.tensor(fake_data).float()
        return fake_data

    real_data = gen_real_dataset(BATCH_SIZE)
    fake_data = gen_fake_dataset(real_data,1)
    w = Wasserstein(real_data, fake_data, Net,BATCH_SIZE,LAMBDA,ITERS)
    w.train()
    w_distance = w.distance()
    print('w_distance:', w_distance)
