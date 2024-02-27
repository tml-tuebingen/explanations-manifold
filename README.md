# The Manifold Hypothesis for Gradient-Based Explanations 

<p align="center">
  <img src="images/landing.png" width="800" alt="Conceptual Overview"/>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?color=g&style=plastic)](https://opensource.org/licenses/MIT)

This is the code repository for the CVPR 2023 [Workshop Paper](http://bit.ly/43SwwbH) "The Manifold Hypothesis for Gradient-Based Explanations" by Sebastian Bordt, Uddeshya Upadhya, Zeynep Akata, and Ulrike von Luxburg. 

## Using the MNIST32 and MNIST256 datasets

We provide the MNIST32 and MNIST256 datasets [here](https://www.kaggle.com/datasets/sbordt/mnist32) and [here](https://www.kaggle.com/datasets/sbordt/mnist256). You can use these datasets in your own research.

We also provide two example notebooks that show how to use the datasets and tangent spaces: [here](https://github.com/tml-tuebingen/explanations-manifold/blob/main/examples/mnist32.ipynb) and [here](https://github.com/tml-tuebingen/explanations-manifold/blob/main/examples/mnist256.ipynb).

You can use these datasets to train models and evaluate the alignment of attributions with the tangent space (or perform any other task that requires the tangent space). Note that the tangent spaces are only provided for the validation images (Certain applications might require the tangent spaces of all the training images. However, we did not require this in our research).

The notebooks in ```replicate-paper/mnist``` show how the datasets were created.

## Replicating the results in our paper

The folder ```replicate-paper``` contains Python files and Jupyter Notebooks that we used to produce the results in our paper. 

- The folder ```replicate-paper/mnist``` contains the files for the MNIST32 and MNIST256 tasks.
- The folder ```replicate-paper/other_datasets``` contains the files for the CIFAR10, EMNIST, Pneumonia and Retinopathy.

  
## Compute tangent spaces and project vectors

We provide code snippets to compute the tangent spaces of generative models and project vectors into the computed tangent spaces.

Compute the tangent space:

```python
def compute_tangent_space(model, z, device="cuda"):
    """Compute the tangent space of a generative model z->x at the given point z.
    The tangent space is spanned by the gradients of the model output with respect to the latent dimensions.
    This function computes these gradients by backpropagation and aggregates the results.

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
```

Project a vector into the computed tangent space:

```python
from scipy.linalg import orth

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
```

## Citing our work

If you use this software in your research, we encourage you to cite our paper.

```bib
@inproceedings{bordt2023manifolds,
  author    = {Sebastian Bordt, Uddeshya Upadhya, Zeynep Akata, and Ulrike von Luxburg},
  title     = {The Manifold Hypothesis for Gradient-Based Explanations},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year      = {2023}
 }
```
