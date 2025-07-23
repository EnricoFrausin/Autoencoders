import matplotlib.pyplot as plt
import numpy as np
import torch



def plot_random_images_and_latents(model, val_loader, device, num_samples=5, EMNIST=False):
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
            img = np.rot90(img, k=1)         # Rotate 90 degrees counterclockwise
            img = np.flipud(img)             # Flip upside down (mirror vertically)
        plt.imshow(img, cmap='gray')
        plt.title(f"Original (idx={idx.item()})")
        plt.axis('off')

        # Latent vector as bar plot
        plt.subplot(2, num_samples, num_samples + i + 1)
        plt.bar(range(latent_vectors.size(1)), latent_vectors[idx].cpu().numpy())
        plt.title("Latent vector")
        plt.xlabel("Feature")
        plt.ylabel("Value")

    plt.tight_layout()
    plt.show()



def visualize_decoded_from_latent(model, num_samples=5, val_loader=None, device=None, EMNIST=False):
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
        plt.imshow(decoded_img, cmap='gray')

        plt.title(f"Decoded idx={idx.item()}")
        plt.axis('off')
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
            img = np.rot90(img, k=1)         # Rotate 90 degrees counterclockwise
            img = np.flipud(img)             # Flip upside down (mirror vertically)
        plt.imshow(img, cmap='gray')

        plt.title(f"Original idx={idx.item()}")
        plt.axis('off')

        # Plot decoded
        plt.subplot(2, num_samples, num_samples + i + 1)


        decoded_img = decoded_img.squeeze()
        if EMNIST == True:
            decoded_img = np.rot90(decoded_img, k=1)  # Rotate 90 degrees counterclockwise
            decoded_img = np.flipud(decoded_img)      # Flip upside down (mirror vertically)

        plt.imshow(decoded_img, cmap='gray')
        plt.title("Decoded")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
