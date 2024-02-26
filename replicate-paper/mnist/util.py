from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset
from torchvision import transforms

from PIL import Image
import seaborn as sns
import numpy as np
import pickle as pkl

import matplotlib.pyplot as plt
from skimage import feature

import numpy as np

from scipy.linalg import orth

from captum.attr import visualization as viz

from captum.attr import Saliency, IntegratedGradients, InputXGradient, NoiseTunnel

from copy import deepcopy
import random


###############################################################################################
# Training Models
###############################################################################################


def zero_one_loss(model, testloader, device="cuda"):
    """get the zero-one loss of a model over a dataloader"""
    model.to(device)
    model.eval()
    test_zero_one_loss = 0
    with torch.no_grad():
        for img, label in testloader:
            img, label = img.to(device), label.to(device)
            pred = model(img)
            test_zero_one_loss += (
                (pred.softmax(dim=1).argmax(dim=1) != label).sum().item()
            )
        test_zero_one_loss = test_zero_one_loss / len(testloader.dataset)
    return test_zero_one_loss


def train(
    model,
    optimizer,
    trainloader,
    testloader,
    device,
    n_epoch=25,
    eps=0.05,
    verbose=True,
):
    """train a model with cross entropy loss"""
    model.to(device)
    ce_loss = torch.nn.CrossEntropyLoss()
    dataiter = trainloader
    if verbose:
        dataiter = tqdm(trainloader)
    for epoch in range(n_epoch):
        # train
        model.train()
        train_loss = 0
        train_zero_one_loss = 0
        for img, label in dataiter:
            img, label = img.to(device), label.to(device)
            pred = model(img)
            optimizer.zero_grad()
            loss = ce_loss(pred, label)
            loss.backward()
            train_loss += loss.item()
            train_zero_one_loss += (
                (pred.softmax(dim=1).argmax(dim=1) != label).sum().item()
            )
            optimizer.step()
        train_zero_one_loss = train_zero_one_loss / len(trainloader.dataset)
        if verbose:
            print("Train Error: ", train_zero_one_loss)
        # test error
        model.eval()
        if testloader is not None:
            test_zero_one_loss = test(model, testloader, device, verbose)
    return train_zero_one_loss, test_zero_one_loss


def test(model, testloader, device, verbose=True):
    """test a models accuracy"""
    model.to(device)
    model.eval()
    test_zero_one_loss = 0
    with torch.no_grad():
        for img, label in testloader:
            img, label = img.to(device), label.to(device)
            pred = model(img)
            test_zero_one_loss += (
                (pred.softmax(dim=1).argmax(dim=1) != label).sum().item()
            )
    test_zero_one_loss = test_zero_one_loss / len(testloader.dataset)
    if verbose:
        print("Test Error: ", test_zero_one_loss)
    return test_zero_one_loss


###############################################################################################
# Estimating the tangent space
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


def to_latent_direction(tangent_space, direction):
    """obtain the direction in latent space that corresponds most closely to a
    direction in ambient space
    """
    dim = tangent_space.shape[0]
    coeff = np.zeros(dim)
    for i in range(dim):
        coeff[i] = tangent_space[i, :, :].flatten() @ direction.flatten()
    return coeff / np.linalg.norm(coeff)


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


def tangent_space_ratio(vectors, tangent_spaces):
    vectors_its = [
        project_into_tangent_space(tangent_spaces[i], vectors[i])
        for i in range(len(vectors))
    ]
    frac_its = [
        np.linalg.norm(vectors_its[i].flatten()) / np.linalg.norm(vectors[i].flatten())
        for i in range(len(vectors))
    ]
    return frac_its


def fraction_in_tangent_space(model, inputs, tangent_spaces):
    # compute gradients
    grad_list, int_grad_list, igx_list = compute_saliency_maps(model, inputs)

    # return the fractions that lie in tangent space
    return (
        tangent_space_ratio(grad_list),
        tangent_space_ratio(int_grad_list),
        tangent_space_ratio(igx_list),
    )


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
# Compute saliency maps
###############################################################################################


def dcn(tensor):
    return tensor.detach().cpu().numpy()


def plain_gradient(model, img, target=None, device="cuda"):
    """NOTE: computation takes place on the given device, but the output will be on the cpu"""
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
    """
    NOTE: computation takes place on the given device, but the output will be on the cpu
    """
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
    """
    NOTE: computation takes place on the given device, but the output will be on the cpu
    """
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
    """
    NOTE: computation takes place on the given device, but the output will be on the cpu
    """
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
    """
    NOTE: computation takes place on the given device, but the output will be on the cpu
    """
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
    """
    NOTE: computation takes place on the given device, but the output will be on the cpu
    """
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


def compute_saliency_maps(model, inputs, targets=None, device="cuda"):
    """for a list of inputs, compute different kinds of saliency maps

    targets: list of targets, same length as inputs or None

    NOTE: computation takes place on the given device, but the output will be on the cpu
    """
    grad_list, int_grad_list, igx_list = [], [], []
    model.eval()
    for idx, img in tqdm(enumerate(inputs)):
        target = targets[idx] if targets is not None else None
        grad_list.append(plain_gradient(model, img, target=target, device=device))
        int_grad_list.append(
            integrated_gradients(model, img, target=target, device=device)
        )
        igx_list.append(input_x_gradient(model, img, target=target, device=device))

    return grad_list, int_grad_list, igx_list


###############################################################################################
# Vizualization and Plotting Saliency Maps
###############################################################################################


def sample_batch(sampler, batchsize, fraction_random_labels=0.0):
    z_batch, x_batch, c_batch = [], [], []
    for i in range(batchsize):
        z, x, c = sampler.sample()
        z_batch.append(z.unsqueeze(0))
        x_batch.append(x.unsqueeze(0).unsqueeze(0))
        c_batch.append(c)
    if fraction_random_labels > 0:
        for idx in range(len(c_batch)):
            if random.random() < fraction_random_labels:
                c_batch[idx] = np.random.randint(10)
    return torch.vstack(z_batch), torch.vstack(x_batch), torch.tensor(c_batch)


def normalize_image(img):
    return (img - img.min()) / (img.max() - img.min())


def plot_saliency_map(axs, attr):
    # attr = np.abs(attr)
    attr = normalize_image(attr)
    axs.imshow(attr, cmap="gray", interpolation="none")


def plot_saliency(ax, smap, vmin, vmax, savefig=None):
    """Plot a saliency map."""
    ax.imshow(smap, vmin=vmin, vmax=vmax, cmap="bwr")
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    if savefig is not None:
        plt.savefig(savefig)


def plot_normalized_saliency(ax, smap, p=0.01, savefig=None):
    """Plot a saliency map with an overlayed image contour."""
    # normalize
    smap = smap / np.linalg.norm(smap.flatten())

    # clip the p-percent larget values off
    quantile = np.quantile(abs(smap.flatten().squeeze()), 1 - p)

    smap[np.where(smap > quantile)] = quantile
    smap[np.where(smap < -quantile)] = -quantile

    # common color space
    v = max(abs(smap.min()), smap.max())
    plot_saliency(ax, smap, -v, v, savefig=savefig)


def plot_saliency_with_image_contour(ax, img, smap, vmin, vmax):
    """Plot a saliency map with an overlayed image contour."""
    img = np.array(1 - img.squeeze())
    img_edges = feature.canny(img, sigma=0.5)
    x = np.zeros((32, 32, 4))
    for i in range(3):
        x[:, :, i] = 1 - img_edges
    x[:, :, 3] = img_edges
    ax.imshow(smap, vmin=vmin, vmax=vmax, cmap="bwr")
    ax.imshow(x, cmap="gray", alpha=0.7)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)


def visualize(
    model,
    images,
    explanations,
    tangent_spaces,
    p=0.005,
    with_contour=True,
    show_predictions=True,
):
    """Visualize the part of an attribution that lies in tagent space and the part of an attribution that is orthogonal to the tangent space.
    The first row shows the original images.
    The second row shows part of attribution that lies in tangent space.
    The third row shows part of attribution that is orthogonal to the tangent space.

    model: The model used to make the predicitons.
    images: A list of original images for which we visualize the attribuitons.
    explanations: A list of attribuitons that should be visualized.
    tangent_spaces: A list of tangent spaces.
    p: The %-tage of absolute values that should be trimmed from the attributions.
    """
    sns.set_style("white")
    fig, axs = plt.subplots(3, 10, figsize=(20, 6))
    for i in range(10):
        img = images[i]
        attr = explanations[i]
        tangent_space = tangent_spaces[i]

        # tangent space decomposition
        attr_in_tangent_space = project_into_tangent_space(tangent_space, attr)
        attr_not_in_tangnet_space = attr - attr_in_tangent_space
        pred = model(img.unsqueeze(0).cuda()).argmax().item()
        ratio = np.linalg.norm(attr_in_tangent_space.flatten()) / np.linalg.norm(
            attr.flatten()
        )

        # normalize vectors
        attr = attr / np.linalg.norm(attr.flatten())
        attr_in_tangent_space = attr_in_tangent_space / np.linalg.norm(
            attr_in_tangent_space.flatten()
        )
        attr_not_in_tangnet_space = attr_not_in_tangnet_space / np.linalg.norm(
            attr_not_in_tangnet_space.flatten()
        )

        # clip the p-percent larget values off
        pooled_vectors = (
            np.array((attr, attr_in_tangent_space, attr_not_in_tangnet_space))
            .flatten()
            .squeeze()
        )
        quantile = np.quantile(abs(pooled_vectors), 1 - p)

        attr[np.where(attr > quantile)] = quantile
        attr[np.where(attr < -quantile)] = -quantile
        attr_in_tangent_space[np.where(attr_in_tangent_space > quantile)] = quantile
        attr_in_tangent_space[np.where(attr_in_tangent_space < -quantile)] = -quantile
        attr_not_in_tangnet_space[np.where(attr_not_in_tangnet_space > quantile)] = (
            quantile
        )
        attr_not_in_tangnet_space[np.where(attr_not_in_tangnet_space < -quantile)] = (
            -quantile
        )

        # common color space
        v = max(
            [
                max(abs(x.min()), x.max())
                for x in [attr, attr_in_tangent_space, attr_not_in_tangnet_space]
            ]
        )

        # the image
        axs[0, i].imshow(1 - img.squeeze(), cmap="gray", interpolation="none")
        plt.setp(axs[0, i].get_xticklabels(), visible=False)
        plt.setp(axs[0, i].get_yticklabels(), visible=False)
        if show_predictions:
            axs[0, i].set_title(f"Prediction: {pred}")

        # saliency maps
        if with_contour:
            plot_saliency_with_image_contour(
                axs[1, i], img, attr_in_tangent_space, -v, v
            )
            plot_saliency_with_image_contour(
                axs[2, i], img, attr_not_in_tangnet_space, -v, v
            )
        else:
            plot_saliency(axs[1, i], attr_in_tangent_space, -v, v)
            plot_saliency(axs[2, i], attr_not_in_tangnet_space, -v, v)
