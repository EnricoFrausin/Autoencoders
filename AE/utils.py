import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import math
from collections import Counter





def get_empirical_states_dict(model, dataloader, verbose=False, threshold_for_binarization=0.5):            # USED IN DEPTH.UTILS
    """
    Extracts binary internal representations from the autoencoder and counts their frequencies.

    Args:
        model: The autoencoder model with sigmoid activation before bottleneck
        dataloader: DataLoader containing the dataset
        device: Device to run computations on

    Returns:
        dict: Dictionary where keys are binary state tuples and values are frequencies
    """
    import torch

    model.eval()
    device = model.device
    state_counts = defaultdict(int)
    total_samples = 0

    with torch.no_grad():
        for batch_data, _ in dataloader:
            batch_data = batch_data.to(device)

            # Encode to get latent representations (after sigmoid)
            latent_vectors = model.encode(batch_data.view(batch_data.size(0), -1))

            # Convert to binary: < 0.5 → 0, >= 0.5 → 1
            binary_states = (latent_vectors >= threshold_for_binarization).int()

            # Convert each binary vector to tuple (hashable for dictionary keys)
            for i in range(binary_states.size(0)):
                state_tuple = tuple(binary_states[i].cpu().numpy())
                state_counts[state_tuple] += 1
                total_samples += 1

    # Normalize frequencies by total_samples
    empirical_states_dict = {k: v / total_samples for k, v in state_counts.items()}

    if verbose:
        print(f"Total samples processed: {total_samples}")
        print(f"Number of unique binary states found: {len(empirical_states_dict)}")
        # print(f"Theoretical maximum states for {model.latent_dim}-dim latent: {2**model.latent_dim}")

    return empirical_states_dict





def analyze_binary_frequencies(empirical_states_dict, top_k=10):
    """
    Analyze and display the most frequent binary states.

    Args:
        empirical_states_dict: Dictionary from get_empirical_states_dict
        top_k: Number of top states to display
    """
    import matplotlib.pyplot as plt

    # Sort by frequency (descending)
    sorted_states = sorted(
        empirical_states_dict.items(), key=lambda x: x[1], reverse=True
    )

    print(f"\nTop {top_k} most frequent binary states:")
    print("-" * 50)

    # Plot frequency distribution
    frequencies = [count for _, count in sorted_states]
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar(range(len(frequencies)), frequencies)
    plt.xlabel("Binary State (sorted by frequency)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Binary States")
    plt.yscale("log")

    plt.subplot(1, 2, 2)
    plt.bar(range(min(top_k, len(frequencies))), frequencies[:top_k])
    plt.xlabel("Top Binary States")
    plt.ylabel("Frequency")
    plt.title(f"Top {top_k} Most Frequent States")

    plt.tight_layout()
    plt.show()

    return sorted_states




# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––




def mean_s_k(n, k, g):  # 0-indexed k
    """
    Calculates the expected value of the k-th feature in a set of n features under the HFM distribution.

    This function computes the mean value for the k-th (0-indexed) feature, given the total number of features and a constant parameter `g` from the HFM distribution. It handles the special case where the parameter xi equals 1 to avoid division by zero.

    Parameters
    ----------
    n : int
        Total number of features.
    k : int
        Index (0-based) of the feature for which the mean is calculated.
    g : float
        Constant parameter in the HFM distribution.

    Returns
    -------
    float
        The expected value (mean) of the k-th feature.
    """

    xi = 2 * np.exp(-g)
    if abs(xi - 1) < 1e-6:
        E = (n - (k + 1) + 2) / (2 * (n + 1))
    else:
        E = 0.5 * (1 + (xi ** (k) - 1) * (xi - 2) / (xi**n + xi - 2))
    return E




def get_m_s(state_tuple, active_category_is_zero=False):            # USED IN DEPTH.UTILS
    """
    Calculates m_s for a given state tuple, 1-indexed.
    m_s is the index of the last active neuron.
    If active_category_is_zero is True, 'active' is represented by 0, the first category.
    If no neuron is active, m_s is 0.
    """
    active_val = 0 if active_category_is_zero else 1
    for i in reversed(range(len(state_tuple))):
        if state_tuple[i] == active_val:
            return i + 1  # 1-indexed
    return 0





def calculate_Z_theoretical(latent_dim, g_param):           # USED IN DEPTH.UTILS
    """
    Calculates the normalization constant Z based on the provided analytical formula.

    Args:
        latent_dim (int): Dimensionality of the latent space.
        g_param (float): The constant 'g'.

    Returns:
        float: The normalization constant Z.
    """
    if math.isclose(g_param, math.log(2)):
        # Handles the case g = log(2) => xi = 1
        Z = 1.0 + np.exp(g_param) * float(latent_dim)  # DIFFERENT FROM PAPER
    else:
        xi = 2.0 * math.exp(-g_param)
        if math.isclose(
            xi, 1.0
        ):  # Should be caught by g = log(2) but good for robustness
            Z = 1.0 + np.exp(-g_param) * float(latent_dim)  # DIFFERENT FROM PAPER
        else:
            sum_geometric_part = (xi**latent_dim - 1.0) / (xi - 1.0)
            Z = 1.0 + np.exp(-g_param) * sum_geometric_part  # DIFFERENT FROM PAPER
    if Z == 0:
        raise ValueError(
            "Calculated theoretical Z is zero, leading to division by zero for probabilities."
        )
    return Z





def get_HFM_prob(m_s: float, g: float, Z: float, logits: True) -> float:
    """
    Calulates the HFM theoretical probability for a state, given m_s, g, and Z.
    If logits=True (default) it returns the log probabilities.
    """
    H_s = m_s - 1  # max(m_s - 1, 0)
    if logits:
        return -g * H_s - np.log(Z)
    return np.exp(-g * H_s) / Z





def calculate_kl_divergence_with_HFM(empirical_states_dict, g):         # USED IN DEPTH.UTILS
    """
    Calculates the KL divergence between an empirical probability distribution
    and a theoretical distribution defined by the HFM with parameter `g`.
    Args:
        empirical_states_dict (dict): A dictionary mapping states (tuples or hashable types) to their empirical probabilities.
        g (float): The parameter of the HFM model controlling the strength of the field.

    Returns:
        float: The calculated KL divergence between the empirical and theoretical distributions.
    Notes:
        - Assumes that the empirical probabilities sum to 1.
    """

    empirical_probs_values = torch.tensor(
        list(empirical_states_dict.values()), dtype=torch.float32
    )
    empirical_distribution = torch.distributions.Categorical(empirical_probs_values)
    empirical_entropy = empirical_distribution.entropy()

    latent_dim = len(next(iter(empirical_states_dict)))
    log_Z = math.log(calculate_Z_theoretical(latent_dim, g))

    mean_H_s = 0

    for state, p_emp in empirical_states_dict.items():
        m_s = get_m_s(state)  # 1-indexed
        mean_H_s += p_emp * m_s

    g_times_H_s = g * mean_H_s

    kl_divergence = -empirical_entropy + g_times_H_s + log_Z

    return kl_divergence
