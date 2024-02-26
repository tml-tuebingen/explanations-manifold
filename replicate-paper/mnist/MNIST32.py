import sys
import os

sys.path.append(os.path.abspath("./disentangling_vae"))
from disvae.utils.modelIO import load_model

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from Simplenet import SimplenetV1


class MNIST32:
    def __init__(
        self,
        modelfile="disentangling_vae/results/btcvae_mnist",
        simplenetfile="./models/mnist_label_simplenet.pth",
        device="cuda",
        Z_DIM=10,
    ):
        self.modelfile = modelfile
        self.simplenetfile = simplenetfile
        self.device = device
        self.Z_DIM = Z_DIM

        self.vae_model = self._load_vae()
        self.classifier = self._load_classifier()

    def sample(self, device="cuda"):
        """main method"""
        p_reject = 1
        while np.random.random() < p_reject:
            prior_sample = torch.randn(1, self.Z_DIM)
            sample = self.vae_model.decode(prior_sample.cuda())
            prediction = self.classifier(sample).softmax(axis=1)
            p_reject = 1 - prediction.max() ** 2
        return (
            prior_sample.flatten(),
            sample.detach().cpu().squeeze(),
            prediction.argmax().item(),
        )

    def _load_vae(self):
        vae_model = load_model(self.modelfile)
        vae_model.eval()
        vae_model.to(self.device)
        return vae_model

    def _load_classifier(self):
        model = SimplenetV1()
        model.load_state_dict(torch.load(self.simplenetfile, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model


class MNIST256(MNIST32):
    def __init__(self, device="cuda"):
        super(MNIST256, self).__init__(device=device)
        self.vae_model32 = self.vae_model
        self.vae_model = InterpolatedDecoder(
            self.vae_model32, size=(256, 256), mode="bilinear"
        )
        self.vae_model.to(self.device)
        self.vae_model.eval()

    def sample(self):
        """main method"""
        p_reject = 1
        while np.random.random() < p_reject:
            prior_sample = torch.randn(1, self.Z_DIM)
            sample = self.vae_model32.decode(
                prior_sample.to(self.device)
            )  # fist decode using the 32x32 model to get the class label
            prediction = self.classifier(sample).softmax(axis=1)
            p_reject = 1 - prediction.max() ** 2
        sample = self.vae_model.decode(
            prior_sample.to(self.device)
        )  # now decode with usampling to get the actual 256x256 sample

        return (
            prior_sample.flatten(),
            sample.detach().cpu().squeeze(),
            prediction.argmax().item(),
        )


class InterpolatedDecoder(nn.Module):
    """Wrapper for a decoder that interpolates (i.e. upsamples) the final image"""

    def __init__(self, decoder, **kwargs):
        super(InterpolatedDecoder, self).__init__()
        self.decoder = decoder
        self.kwargs = kwargs

    def forward(self, x):
        raise NotImplementedError("InterpolatedDecoder does not support forward()")

    def decode(self, z):
        x = self.decoder.decode(z)
        return F.interpolate(x, **self.kwargs)

    def sample(self, num_samples, device="cuda"):
        z, x = self.decoder.sample(num_samples, device=device)
        x = F.interpolate(x, **self.kwargs)
        return z, x
