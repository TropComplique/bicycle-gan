import json
import torch
import torch.nn.init as init
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from input_pipeline import PairsDataset
from networks.generator import Generator
from networks.encoder import ResNetEncoder

BATCH_SIZE = 8
DATA = '/home/dan/datasets/edges2shoes/train/'
NUM_EPOCHS = 50
DEVICE = torch.device('cuda:0')
MODEL_SAVE_PREFIX = 'models/run00_just_ae'
TRAIN_LOGS = 'losses_run00_just_ae.json'
SAVE_STEP = 20000


class AE:

    def __init__(self, device, num_steps, z_dimension=8):

        # in and out channels for the generator:
        a, b = 2, 3

        G = Generator(a, b)
        E = ResNetEncoder(b, z_dimension)

        def weights_init(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.xavier_normal_(m.weight, gain=0.02)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d) and m.affine:
                init.ones_(m.weight)
                init.zeros_(m.bias)

        self.G = G.apply(weights_init).to(device)
        self.E = E.apply(weights_init).to(device)
        self.device = device

        params = {
            'lr': 1e-3,
            'betas': (0.5, 0.999),
            'weight_decay': 1e-8
        }
        generator_groups = [
            {'params': [p for n, p in self.G.named_parameters() if 'mapping' not in n]},
            {'params': self.G.mapping.parameters(), 'lr': 1e-4}
        ]
        self.optimizer = {
            'G': optim.Adam(generator_groups, **params),
            'E': optim.Adam(self.E.parameters(), **params)
        }

        def lambda_rule(i):
            decay = num_steps // 2
            m = 1.0 if i < decay else 1.0 - (i - decay) / decay
            return max(m, 0.0)

        self.schedulers = []
        for o in self.optimizer.values():
            self.schedulers.append(LambdaLR(o, lr_lambda=lambda_rule))

    def train_step(self, A, B):
        """
        The input tensors represent images
        with pixel values in [0, 1] range.

        Arguments:
            A: a float tensor with shape [n, a, h, w].
            B: a float tensor with shape [n, b, h, w].
        Returns:
            a dict with float numbers.
        """
        A = A.to(self.device)
        B = B.to(self.device)

        # batch size
        n = A.size(0)

        mean, logvar = self.E(B)
        std = logvar.mul(0.5).exp()
        z = torch.randn(n, 8, device=self.device)
        # they all have shape [n, z_dimension]

        B_restored = self.G(A, mean + z * std)  # shape [n, b, h, w]
        l1_loss = F.l1_loss(B_restored, B)

        kl_loss = 0.5 * (logvar.exp() + mean.pow(2) - 1.0 - logvar).sum(1).mean(0)
        total_loss = l1_loss + 5e-3 * kl_loss

        self.optimizer['G'].zero_grad()
        self.optimizer['E'].zero_grad()
        total_loss.backward()
        self.optimizer['G'].step()
        self.optimizer['E'].step()

        # decay the learning rate
        for s in self.schedulers:
            s.step()

        loss_dict = {
            'l1_loss': l1_loss.item(),
            'kl_loss': kl_loss.item()
        }
        return loss_dict

    def save_model(self, model_path):
        torch.save(self.G.state_dict(), model_path + '_generator.pth')
        torch.save(self.E.state_dict(), model_path + '_encoder.pth')


def main():

    dataset = PairsDataset(folder=DATA, size=256, is_training=True)
    data_loader = DataLoader(
        dataset=dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True
    )
    num_steps = NUM_EPOCHS * (len(dataset) // BATCH_SIZE)
    model = AE(device=DEVICE, num_steps=num_steps)

    logs = []
    i = 0  # number of weight updates
    text = 'e: {0}, i: {1}, l1_loss: {2:.5f}, kl_loss: {3:.6f}'

    for e in range(NUM_EPOCHS):
        for A, B in data_loader:

            i += 1
            losses = model.train_step(A, B)

            log = text.format(e, i, losses['l1_loss'], losses['kl_loss'])
            print(log)
            logs.append(losses)

            if i % SAVE_STEP == 0:
                model.save_model(MODEL_SAVE_PREFIX)
                with open(TRAIN_LOGS, 'w') as f:
                    json.dump(logs, f)


main()
