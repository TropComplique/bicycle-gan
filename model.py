import torch
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F


class LSGAN(nn.Module):

    def __init__(self):
        super(LSGAN, self).__init__()

    def forward(self, x, is_real):
        """
        Arguments:
            x: a float tensor with any shape.
            is_real: a boolean.
        Returns:
            a float tensor with shape [].
        """
        if is_real:
            return torch.pow(x - 1.0, 2).mean()

        return torch.pow(x, 2).mean()


class BicycleGAN:

    def __init__(self, device, z_dimension=8):

        # in and out channels for the generator:
        a, b = 1, 3

        self.G = UNet(a, b, depth=64, downsample=8)
        self.E = ResNetEncoder(a, z_dimension, depth=64, num_blocks=5)
        self.D1 = MultiScaleDiscriminator(b, depth=64)
        self.D2 = MultiScaleDiscriminator(b, depth=64)

        def weights_init(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                torch.nn.init.xavier_normal_(m.weight, gain=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.G = G.apply(weights_init).to(device)
        self.E = E.apply(weights_init).to(device)
        self.D1 = D1.apply(weights_init).to(device)
        self.D2 = D2.apply(weights_init).to(device)

        betas = (0.5, 0.999)
        self.G_optimizer = optim.Adam(lr=2e-4, params=self.G.parameters(), betas=betas)
        self.E_optimizer = optim.Adam(lr=2e-4, params=self.E.parameters(), betas=betas)
        self.D1_optimizer = optim.Adam(lr=4e-4, params=self.D1.parameters(), betas=betas)
        self.D2_optimizer = optim.Adam(lr=4e-4, params=self.D2.parameters(), betas=betas)

        self.gan_loss = LSGAN()
        self.z_dimension = z_dimension
        self.device = device

    def get_random(self, b):
        return torch.randn(b, self.z_dimension, device=self.device)

    def train_step(self, A, B):
        """
        Arguments:
            A: a float tensor with shape [n, a, h, w].
            B: a float tensor with shape [2 * n, b, h, w].
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
        kl_loss = 0.5 * (logvar.exp() + mean.pow(2) - 1.0 - logvar).sum(0)

        # UPDATE GENERATOR AND ENCODER

        total_loss = fool_d1_loss + fool_d2_loss + 10.0 * l1_loss + 1e-2 * kl_loss
        self.G_optimizer.zero_grad()
        self.E_optimizer.zero_grad()
        total_loss.backward()
        self.G_optimizer.step()
        self.E_optimizer.step()

        # LATENT REGRESSION LOSS

        lr_loss = F.l1_loss(mu, z)

        # UPDATE THE GENERATOR ONLY

        self.G_optimizer.zero_grad()
        lr_loss.backward()
        self.G_optimizer.step()

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

        self.D1_optimizer.zero_grad()
        self.D2_optimizer.zero_grad()
        d_loss.backward()
        self.D1_optimizer.step()
        self.D2_optimizer.step()

        loss_dict = {
            'fool_d1_loss': fool_d1_loss.item(),
            'fool_d2_loss': fool_d2_loss.item(),
            'l1_loss': l1_loss.item(),
            'kl_loss': kl_loss.item(),
            'lr_loss': lr_loss.item(),
            'discriminators_loss': d_loss.item(),
        }
        return loss_dict

    def save_model(self, model_path):
        torch.save(self.G.state_dict(), model_path + '_generator.pth')
        torch.save(self.E.state_dict(), model_path + '_encoder.pth')
        torch.save(self.D1.state_dict(), model_path + '_discriminator1.pth')
        torch.save(self.D2.state_dict(), model_path + '_discriminator2.pth')
