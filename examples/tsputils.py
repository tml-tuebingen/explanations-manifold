"""
Title: tsputils.py

Description: Compute the tangent space of generative models and compute saliency maps for images.

Author: Sebastian Bordt
Date: 2024
Version: 0.1.0

License: MIT License. 
"""

import numpy as np

import numpy as np

from scipy.linalg import orth

from captum.attr import (
    Saliency,
    IntegratedGradients,
    InputXGradient,
    NoiseTunnel,
)

from copy import deepcopy


###############################################################################################
# Tangent space
###############################################################################################


def compute_tangent_space(model, z, device="cuda"):
    """Compute the tangent space of a generative model z->x at the given point z.
    The tangent space is spanned by the gradients of the model output with respect to the latent dimensions.
    This function computes these gradients by backpropagaion and aggregates the results.

    Parameters:
    - model (pytorch module): The generative model. A pytorch module that implements decode(z).
    - z (pytorch tensor): The latent vector. Has to be 1-dimensional: Batch dimension in z is not supported.
    - device (str): The device on which the computation should take place.

    Returns:
    - pytorch tensor: Vectors that span the tangent space (tangent space dim x model output dim). The ordering of the returned vectors corresponds to the latent dimensions of z.
    """
    assert (
        len(z.shape) == 1
    ), "compute_tangent_space: z has to be a 1-dimensional vector: Batch dimension in z is not supported."
    model.to(device)
    z = z.to(device)
    latent_dim = z.shape[0]
    z.requires_grad = True
    out = model.decode(z)
    out = out.squeeze()  # remove singleton batch dimension
    output_shape = out.shape  # store original output shape
    out = out.reshape(-1)  # transform the output into a vector
    tangent_space = torch.zeros((latent_dim, out.shape[0]))
    for i in range(out.shape[0]):
        out[i].backward(retain_graph=True)
        tangent_space[:, i] = z.grad
        z.grad.zero_()
    tangent_space = tangent_space.reshape(
        (-1, *output_shape)
    )  # tangent space in model output shape
    return tangent_space


def project_into_tangent_space(tangent_space, vector):
    """Compute the projection of a vector into the tangent space.

    Parameters:
    - tangent space (pytorch tensor): The tangent space. Expected dimension: [tangent_space_dimension, IMG_DIM, IMG_DIM]
    - vector (pytorch tensor): The vector. Expected dimension: [IMG_DIM, IMG_DIM]

    Returns:
    - pytorch tensor: The projection of the vector into the tangent space.
    """
    IMG_DIM = tangent_space.shape[-1]
    tangent_space_orth = orth(
        tangent_space.reshape((-1, IMG_DIM * IMG_DIM)).T
    ).T.reshape((-1, IMG_DIM, IMG_DIM))
    dim = tangent_space_orth.shape[0]
    coeff = np.zeros(dim)
    for i in range(dim):
        coeff[i] = tangent_space_orth[i, :, :].flatten() @ vector.flatten()
    vector_in_tangent_space = (coeff @ tangent_space_orth.reshape((dim, -1))).reshape(
        (IMG_DIM, IMG_DIM)
    )
    return vector_in_tangent_space


def tangent_space_per_pixel(tangent_space):
    """For each pixel, i.e. each unit vector in a img_dim*img_dim-dimensional space, compute the fraction in tangent space
    and return the result as an image.

    tangent_space: array or tensor of shape (tangent_space_dimension, img_dim, img_dim)

    returns: array or tensor of shape (img_dim, img_dim)
    """
    assert (
        len(tangent_space.shape) == 3
        and tangent_space.shape[1] == tangent_space.shape[2]
    )
    img_dim = tangent_space.shape[1]
    tangent_space_orth = orth(
        tangent_space.reshape((-1, img_dim * img_dim)).T
    ).T.reshape((-1, img_dim, img_dim))
    return np.square(tangent_space_orth).sum(axis=0)


###############################################################################################
# Saliency maps
###############################################################################################


def dcn(tensor):
    return tensor.detach().cpu().numpy()


def plain_gradient(model, img, target=None, device="cuda"):
    model.eval()
    if len(img.shape) == 2:  # expand to 4 dimensions
        img = img.unsqueeze(0)
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    img = img.to(device)
    img.requires_grad_(True)
    if img.grad is not None:
        img.grad.zero_()
    pred = model(img)
    if target is None:  # predicted class
        target = pred[0].argmax().item()
    pred[0][target].backward()
    grad = deepcopy(dcn(img.grad).squeeze())
    return grad


def smooth_grad(model, img, std, n_samples, target=None, device="cuda"):
    model.eval()
    if len(img.shape) == 2:  # expand to 4 dimensions
        img = img.unsqueeze(0)
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    img = img.to(device)
    img.requires_grad_(True)
    if img.grad is not None:
        img.grad.zero_()
    if target is None:  # predicted class
        pred = model(img)
        target = pred[0].argmax().item()
    grad = Saliency(model)
    noise_tunnel = NoiseTunnel(grad)
    smoothed_grad = noise_tunnel.attribute(
        img,
        nt_samples=n_samples,
        stdevs=std,
        nt_type="smoothgrad",
        target=target,
        abs=False,
    )
    smoothed_grad = deepcopy(dcn(smoothed_grad).squeeze())
    return smoothed_grad


def integrated_gradients(model, img, target=None, device="cuda"):
    model.eval()
    if len(img.shape) == 2:  # expand to 4 dimensions
        img = img.unsqueeze(0)
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    img = img.to(device)
    img.requires_grad_(True)
    if img.grad is not None:
        img.grad.zero_()
    if target is None:  # predicted class
        pred = model(img)
        target = pred[0].argmax().item()
    ig = IntegratedGradients(model, multiply_by_inputs=False)
    int_grad, _ = ig.attribute(img, target=target, return_convergence_delta=True)
    int_grad = deepcopy(dcn(int_grad).squeeze())
    return int_grad


def smooth_integrated_gradients(model, img, std, n_samples, target=None, device="cuda"):
    model.eval()
    if len(img.shape) == 2:  # expand to 4 dimensions
        img = img.unsqueeze(0)
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    img = img.to(device)
    img.requires_grad_(True)
    if img.grad is not None:
        img.grad.zero_()
    if target is None:  # predicted class
        pred = model(img)
        target = pred[0].argmax().item()
    ig = IntegratedGradients(model)
    noise_tunnel = NoiseTunnel(ig)
    smoothed_int_grad = noise_tunnel.attribute(
        img, nt_samples=n_samples, stdevs=std, nt_type="smoothgrad", target=target
    )
    smoothed_int_grad = deepcopy(dcn(smoothed_int_grad).squeeze())
    return smoothed_int_grad


def input_x_gradient(model, img, target=None, device="cuda"):
    model.eval()
    if len(img.shape) == 2:  # expand to 4 dimensions
        img = img.unsqueeze(0)
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    img = img.to(device)
    img.requires_grad_(True)
    if img.grad is not None:
        img.grad.zero_()
    if target is None:  # predicted class
        pred = model(img)
        target = pred[0].argmax().item()
    ixg = InputXGradient(model)
    attr_ixg = ixg.attribute(img, target=target)
    attr_ixg = deepcopy(dcn(attr_ixg).squeeze())
    return attr_ixg


def smooth_input_x_gradient(model, img, std, n_samples, target=None, device="cuda"):
    model.eval()
    if len(img.shape) == 2:  # expand to 4 dimensions
        img = img.unsqueeze(0)
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    img = img.to(device)
    img.requires_grad_(True)
    if img.grad is not None:
        img.grad.zero_()
    if target is None:  # predicted class
        pred = model(img)
        target = pred[0].argmax().item()
    ixg = InputXGradient(model)
    noise_tunnel = NoiseTunnel(ixg)
    smoothed_ixg = noise_tunnel.attribute(
        img, nt_samples=n_samples, stdevs=std, nt_type="smoothgrad", target=target
    )
    smoothed_ixg = deepcopy(dcn(smoothed_ixg).squeeze())
    return smoothed_ixg
