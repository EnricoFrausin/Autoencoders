import matplotlib.pyplot as plt
import numpy as np
import torch

from AE.utils import calc_hfm_kld
from AE.utils import calc_Z_theoretical, calc_hfm_prob
from AE.utils import calc_ms




def save_fig(save_dir, title):

    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, title)
        plt.savefig(save_path)

    return


def plot_random_images_and_latents(
    model, val_loader, device, num_samples=5, EMNIST=False
):
    """
    Plots random original images from the validation loader and their corresponding latent vectors as bar plots.
    """
    images, labels = next(iter(val_loader))
    images = images.to(device)

    with torch.no_grad():
        latent_vectors = model.encode(images.view(images.size(0), -1))

    indices = torch.randint(0, images.size(0), (num_samples,))

    plt.figure(figsize=(num_samples * 3, 6))
    for i, idx in enumerate(indices):
        # Original image
        plt.subplot(2, num_samples, i + 1)
        img = images[idx].cpu().squeeze().numpy()
        if EMNIST == True:
            img = np.rot90(img, k=1)  # Rotate 90 degrees counterclockwise
            img = np.flipud(img)  # Flip upside down (mirror vertically)
        plt.imshow(img, cmap="gray")
        plt.title(f"Original (idx={idx.item()})")
        plt.axis("off")

        # Latent vector as bar plot
        plt.subplot(2, num_samples, num_samples + i + 1)
        plt.bar(range(latent_vectors.size(1)), latent_vectors[idx].cpu().numpy())
        plt.title("Latent vector")
        plt.xlabel("Feature")
        plt.ylabel("Value")

    plt.tight_layout()
    plt.show()





def visualize_decoded_from_latent(
    model, num_samples=5, val_loader=None, device=None, EMNIST=False
):
    """
    Samples random latent vectors from the internal representation and plots the decoded images.
    """
    if val_loader is None or model is None or device is None:
        print("Please provide val_loader, model, and device.")
        return

    # Get a batch of images from the validation loader
    images, labels = next(iter(val_loader))
    images = images.to(device)

    # Pass images through the encoder to get latent vectors
    with torch.no_grad():
        latent_vectors = model.encode(images.view(images.size(0), -1))

    # Select random indices to visualize
    indices = torch.randint(0, images.size(0), (num_samples,))

    plt.figure(figsize=(num_samples * 3, 3))
    for i, idx in enumerate(indices):
        latent = latent_vectors[idx].unsqueeze(0)
        # Decode the latent vector to get the reconstructed image
        with torch.no_grad():
            decoded = model.decode(latent)
        decoded_img = decoded.cpu().numpy().reshape(images.shape[1:])
        plt.subplot(1, num_samples, i + 1)

        decoded_img = decoded_img.squeeze()
        if EMNIST == True:
            decoded_img = np.rot90(decoded_img, k=1)
            decoded_img = np.flipud(decoded_img)  # Flip upside down (mirror vertically)
        plt.imshow(decoded_img, cmap="gray")

        plt.title(f"Decoded idx={idx.item()}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()





def plot_original_vs_decoded(model, data_loader, device, num_samples=5, EMNIST=False):
    """
    Samples images from the dataset, encodes and decodes them, and plots original vs decoded images side by side.
    """

    # Get a batch of images from the validation loader
    images, labels = next(iter(data_loader))
    images = images.to(device)

    # Select random indices to visualize
    indices = torch.randint(0, images.size(0), (num_samples,))

    plt.figure(figsize=(num_samples * 4, 4))
    for i, idx in enumerate(indices):
        img = images[idx].unsqueeze(0)
        # Flatten and encode
        with torch.no_grad():
            latent = model.encode(img.view(1, -1))
            decoded = model.decode(latent)
        decoded_img = decoded.cpu().numpy().reshape(img.shape[1:])

        # Plot original
        plt.subplot(2, num_samples, i + 1)

        img = images[idx].cpu().squeeze().numpy()
        if EMNIST == True:
            img = np.rot90(img, k=1)  # Rotate 90 degrees counterclockwise
            img = np.flipud(img)  # Flip upside down (mirror vertically)
        plt.imshow(img, cmap="gray")

        plt.title(f"Original idx={idx.item()}")
        plt.axis("off")

        # Plot decoded
        plt.subplot(2, num_samples, num_samples + i + 1)

        decoded_img = decoded_img.squeeze()
        if EMNIST == True:
            decoded_img = np.rot90(
                decoded_img, k=1
            )  # Rotate 90 degrees counterclockwise
            decoded_img = np.flipud(decoded_img)  # Flip upside down (mirror vertically)

        plt.imshow(decoded_img, cmap="gray")
        plt.title("Decoded")
        plt.axis("off")

    plt.tight_layout()
    plt.show()





# ––––––––––––––––––––––––––––––––––––––OPTIMAL G PARAMETER PLOTS–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––



def plot_expected_ms_vs_g(gauged_states, g_range=np.linspace(0.1, 5.0, 50)):
    """
    Plots the expected m_s value for a range of g values.

    Args:
        gauged_states (dict): Dictionary mapping state tuples to their empirical probabilities.
        g_range (array-like): Sequence of g values to evaluate.
    """
    latent_dim = len(next(iter(gauged_states)))
    m_s_values = {
        state: get_m_s(state, active_category_is_zero=False)
        for state in gauged_states.keys()
    }
    expected_ms_list = []

    for g in g_range:
        Z = calculate_Z_theoretical(latent_dim, g)
        expected_ms = 0.0
        for state, empirical_prob in gauged_states.items():
            m_s = m_s_values[state]
            hfm_prob = get_HFM_prob(m_s, g, Z, logits=False)
            # expected_ms += m_s * hfm_prob * empirical_prob
            expected_ms += m_s * hfm_prob
        # expected_ms += m_s * empirical_prob
        expected_ms_list.append(expected_ms)

    plt.figure(figsize=(8, 6))
    plt.plot(g_range, expected_ms_list, marker="o")
    plt.xlabel("g")
    plt.ylabel("Expected m_s")
    plt.title("Expected m_s vs g")
    plt.grid(True, alpha=0.3)
    plt.show()





def plot_expected_kl_vs_g(gauged_states, g_range=np.linspace(0.1, 5.0, 50)):
    """
    Plots the expected Kullback-Leibler (KL) divergence as a function of the parameter `g`.

    Args:
        gauged_states: The input data or states for which the KL divergence is to be calculated.
        g_range (array-like, optional): The range of `g` values over which to compute and plot the expected KL divergence.
            Defaults to a numpy array of 50 values linearly spaced between 0.1 and 5.0.

    Returns:
        None: This function displays a plot and does not return any value.
    """
    expected_kl_list = []

    for g in g_range:
        expected_kl_list.append(calculate_kl_divergence_with_HFM(gauged_states, g))

    plt.figure(figsize=(8, 6))
    plt.plot(g_range, expected_kl_list, marker="o")
    plt.xlabel("g")
    plt.ylabel("Expected KL Divergence")
    plt.title("Expected KL Divergence vs g")
    plt.grid(True, alpha=0.3)
    plt.show()




def visualize_bottleneck_neurons(model, device, img_shape=(28, 28), save_dir = None, file_name=None):
    """
    Visualize what each feature in the bottleneck represents by decoding one-hot vectors.
    
    Args:
        model: The trained autoencoder model with a .decode() method.
        device: The device to run computations on.
        img_shape: Shape to reshape the decoded output (default: (28, 28) for MNIST).
    """
    model.eval()
    latent_dim = model.latent_dim
    n_cols = (latent_dim + 1) // 2
    n_rows = 2
    with torch.no_grad():
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 5))
        axes = axes.flatten()
        for i in range(latent_dim):
            one_hot = torch.zeros((1, latent_dim), device=device)
            one_hot[0, i] = 1.0
            decoded = model.decode(one_hot).cpu().view(*img_shape)
            ax = axes[i]
            ax.imshow(decoded, cmap='gray')
            ax.set_title(f'Neuron {i+1}')
            ax.axis('off')
        # Hide any unused subplots
        for j in range(latent_dim, n_rows * n_cols):
            axes[j].axis('off')
        plt.tight_layout()

        save_fig(save_dir, file_name)

        plt.show()




#–––––––––––––––––––––––––––––––––––––––––DEPTH ANALYSIS–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––



def plot_KLs_vs_hidden_layers(KLs, gs, dataset_name, save_dir = None):                                  # EXPORTED TO DEPTH_ANALYSIS
    """
    Plots KLs vs number of hidden layers, with gs indicated by a colormap.
    Assumes KLs and gs are lists of length 4 (for 1 to 4 hidden layers).
    """

    num_layers = np.arange(1, len(KLs) + 1)
    gs = np.array(gs)

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(num_layers, KLs, c=gs, cmap="viridis", s=100)
    plt.colorbar(scatter, label="g")
    plt.xlabel("Number of Hidden Layers")
    plt.ylabel("KL Divergence")
    plt.title("KL Divergence vs Hidden Layers - "+ f"\n{dataset_name} Dataset")
    plt.xticks(num_layers)
    plt.grid(True)

    save_fig(save_dir, f"KL_{dataset_name}.png")

    plt.show()


def datasets_dicts_comparison(KLs_dict, save_dir = None):                                            # EXPORTED TO DEPTH_ANALYSIS
    """
    Plots KL values for each dataset in KLs_dict on the same graph.
    X-axis: number of hidden layers (1, 2, 3, ...)
    """
    plt.figure(figsize=(8, 5))
    for key, values in KLs_dict.items():
        x = list(range(1, len(values) + 1))
        plt.plot(x, values, marker='o', label=key)
    plt.xlabel("Number of hidden layers")
    plt.ylabel("KL value")
    plt.title("KLs vs Number of Hidden Layers")
    plt.legend()
    plt.grid(True)

    save_dir(save_dir, "comparison.png")
    
    plt.show()
