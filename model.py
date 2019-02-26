import torch
import torch.nn.init as init
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from networks.unet import UNet
from networks.encoder import ResNetEncoder
from networks.discriminators import MultiScaleDiscriminator


class LSGAN(nn.Module):

    def __init__(self):
        super(LSGAN, self).__init__()

    def forward(self, x, is_real):
        """
        Arguments:
            x: a tuple of float tensors with any shape.
            is_real: a boolean.
        Returns:
            a float tensor with shape [].
        """
        x1, x2 = x

        if is_real:
            return torch.pow(x1 - 1.0, 2).mean() + torch.pow(x2 - 1.0, 2).mean()

        return torch.pow(x1, 2).mean() + torch.pow(x2, 2).mean()


class BicycleGAN:

    def __init__(self, device, num_steps, z_dimension=8):

        # in and out channels for the generator:
        a, b = 1, 3

        G = UNet(a, b, depth=64)
        E = ResNetEncoder(b, z_dimension, depth=64)
        D1 = MultiScaleDiscriminator(b, depth=64, downsample=3)
        D2 = MultiScaleDiscriminator(b, depth=64, downsample=3)

        def weights_init(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.xavier_normal_(m.weight, gain=0.02)
                if m.bias is not None:
                    init.zeros_(m.bias)

        self.G = G.apply(weights_init).to(device)
        self.E = E.apply(weights_init).to(device)
        self.D1 = D1.apply(weights_init).to(device)
        self.D2 = D2.apply(weights_init).to(device)

        betas = (0.5, 0.999)
        self.optimizer = {
            'G': optim.Adam(lr=2e-4, params=self.G.parameters(), betas=betas),
            'E': optim.Adam(lr=2e-4, params=self.E.parameters(), betas=betas),
            'D1': optim.Adam(lr=4e-4, params=self.D1.parameters(), betas=betas),
            'D2': optim.Adam(lr=4e-4, params=self.D2.parameters(), betas=betas)
        }

        self.schedulers = []
        for o in self.optimizer.values():
            self.schedulers.append(CosineAnnealingLR(o, T_max=num_steps, eta_min=1e-7))

        self.gan_loss = LSGAN()
        self.z_dimension = z_dimension
        self.device = device

    def get_random(self, b):
        return torch.randn(b, self.z_dimension, device=self.device)

    def train_step(self, A, B):
        """
        The input tensors represent images
        with pixel values in [0, 1] range.

        Arguments:
            A: a float tensor with shape [n, a, h, w].
            B: a float tensor with shape [2 * n, b, h, w].
        Returns:
            a dict with float numbers.
        """
        A = A.to(self.device)
        B = B.to(self.device)

        n = A.size(0)  # batch size
        B, B_another = torch.split(B, [n, n], dim=0)

        # ENCODE AND THEN SYNTHESIZE

        mean, logvar = self.E(B)
        std = logvar.mul(0.5).exp()
        # they all have shape [n, z_dimension]

        z = self.get_random(n)  # shape [n, z_dimension]
        B_restored = self.G(A, mean + z * std)  # shape [n, b, h, w]

        # SYNTHESIZE AND THEN ENCODE

        z = self.get_random(n)  # shape [n, z_dimension]
        B_generated = self.G(A, z)  # shape [n, b, h, w]
        mu, _ = self.E(B_generated)

        # FREEZE THE DISCRIMINATORS

        for p in self.D1.parameters():
            p.requires_grad = False
        for p in self.D2.parameters():
            p.requires_grad = False

        # FOOL THE DISCRIMINATORS LOSSES

        scores1 = self.D1(B_restored)
        fool_d1_loss = self.gan_loss(scores1, True)

        scores2 = self.D2(B_generated)
        fool_d2_loss = self.gan_loss(scores2, True)

        # RECONSTRUCTION LOSS

        l1_loss = F.l1_loss(B_restored, B)

        # KL DIVERGENCE

        # note that KL(N(mean, var) || N(0, 1)) =
        # = 0.5 * sum_k (exp(log(var_k)) + mean_k^2 - 1 - log(var_k)),
        # where mean and var are k-dimensional vectors
        kl_loss = 0.5 * (logvar.exp() + mean.pow(2) - 1.0 - logvar).sum(1).mean(0)

        # UPDATE GENERATOR AND ENCODER

        total_loss = fool_d1_loss + fool_d2_loss + 10.0 * l1_loss + 1e-2 * kl_loss
        self.optimizer['G'].zero_grad()
        self.optimizer['E'].zero_grad()
        total_loss.backward(retain_graph=True)
        self.optimizer['G'].step()
        self.optimizer['E'].step()

        # LATENT REGRESSION LOSS

        lr_loss = F.l1_loss(mu, z)

        # UPDATE THE GENERATOR ONLY

        self.optimizer['G'].zero_grad()
        lr_loss.backward()
        self.optimizer['G'].step()

        # UPDATE THE DISCRIMINATORS

        for p in self.D1.parameters():
            p.requires_grad = True
        for p in self.D2.parameters():
            p.requires_grad = True

        d_loss = torch.tensor(0.0, device=self.device)

        scores = self.D1(B_restored.detach())
        d_loss += self.gan_loss(scores, False)

        scores = self.D2(B_generated.detach())
        d_loss += self.gan_loss(scores, False)

        scores = self.D1(B)
        d_loss += self.gan_loss(scores, True)

        scores = self.D2(B_another)
        d_loss += self.gan_loss(scores, True)

        self.optimizer['D1'].zero_grad()
        self.optimizer['D2'].zero_grad()
        d_loss.backward()
        self.optimizer['D1'].step()
        self.optimizer['D2'].step()

        # decay learning rate
        for s in self.schedulers:
            s.step()

        loss_dict = {
            'fool_d1_loss': fool_d1_loss.item(),
            'fool_d2_loss': fool_d2_loss.item(),
            'l1_loss': l1_loss.item(),
            'kl_loss': kl_loss.item(),
            'lr_loss': lr_loss.item(),
            'total_loss': total_loss.item(),
            'discriminators_loss': d_loss.item(),
        }
        return loss_dict

    def save_model(self, model_path):
        torch.save(self.G.state_dict(), model_path + '_generator.pth')
        torch.save(self.E.state_dict(), model_path + '_encoder.pth')
        torch.save(self.D1.state_dict(), model_path + '_discriminator1.pth')
        torch.save(self.D2.state_dict(), model_path + '_discriminator2.pth')
