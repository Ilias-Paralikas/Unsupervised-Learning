import torch
import torch.nn as nn
import torch.nn.functional as F


class WGANGPLoss:
    """
    Wasserstein GAN with gradient penalty.

    Usage
    -----
    critic step (N_CRITIC times):
        loss = wgan_gp.critic_loss(disc, real, fake.detach())
        loss.backward(); disc_opt.step()

    generator step (once):
        loss = wgan_gp.generator_loss(disc, fake)   # fake NOT detached
        loss.backward()
    """

    def __init__(self, lambda_gp: float = 10.0, drift_weight: float = 0.001):
        self.lambda_gp    = lambda_gp
        self.drift_weight = drift_weight

    def gradient_penalty(
        self,
        disc:          nn.Module,
        real:          torch.Tensor,  # (K, C, P, P)  detached
        fake_detached: torch.Tensor,  # (K, C, P, P)  detached
    ) -> torch.Tensor:
        K = real.shape[0]
        alpha = torch.rand(K, 1, 1, 1, device=real.device)
        interp = (alpha * real + (1.0 - alpha) * fake_detached).requires_grad_(True)

        d_interp = disc(interp)
        grads = torch.autograd.grad(
            outputs=d_interp,
            inputs=interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True,
            retain_graph=True,
        )[0]                                          # (K, C, P, P)

        gp = ((grads.norm(2, dim=(1, 2, 3)) - 1) ** 2).mean()
        return self.lambda_gp * gp

    def critic_loss(
        self,
        disc:          nn.Module,
        real:          torch.Tensor,  # (K, C, P, P)
        fake_detached: torch.Tensor,  # (K, C, P, P)  must be detached by caller
    ) -> torch.Tensor:
        d_real = disc(real)
        d_fake = disc(fake_detached)
        gp     = self.gradient_penalty(disc, real, fake_detached)
        drift  = self.drift_weight * d_real.pow(2).mean()
        return d_fake.mean() - d_real.mean() + gp + drift

    def generator_loss(
        self,
        disc: nn.Module,
        fake: torch.Tensor,  # (K, C, P, P)  grad graph intact
    ) -> torch.Tensor:
        return -disc(fake).mean()

    def feature_matching_loss(
        self,
        disc: nn.Module,
        real: torch.Tensor,  # (K, C, P, P)  positionally aligned with fake
        fake: torch.Tensor,  # (K, C, P, P)  grad graph intact
    ) -> torch.Tensor:
        """
        MSE between discriminator intermediate features of the real patch and
        the reconstructed patch at the same position.  The discriminator still
        trains normally with critic_loss; only the generator-side loss changes.

        Avoids the real/fake minimax game for the generator — no mode collapse,
        no competing objectives — while still using the discriminator's learned
        domain-specific features to enforce texture quality.
        """
        with torch.no_grad():
            feat_real = disc.get_features(real)

        feat_fake = disc.get_features(fake)

        loss = sum(
            F.mse_loss(ff, fr)
            for ff, fr in zip(feat_fake, feat_real)
        )
        return loss / len(feat_real)
