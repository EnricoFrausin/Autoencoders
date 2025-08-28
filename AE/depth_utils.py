import numpy as np
from itertools import permutations, combinations
import os
import matplotlib.pyplot as plt
import random
import math

from AE.utils import calculate_kl_divergence_with_HFM
from AE.utils import get_empirical_states_dict
from AE.utils import calculate_Z_theoretical
from AE.utils import get_m_s



#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––



def flip_gauge_bits(empirical_states_dict):  # for return_minimum_kl
    """
    Flip specific bits in all states based on the activated bits in the most frequent state.

    Args:
        empirical_states_dict (dict): Dictionary mapping state tuples to their empirical probabilities.
                                From get_empirical_states_dict.

    Returns:
        dict: A new dictionary with the same probabilities but flipped states according to the rule.
    """
    if not empirical_states_dict:
        return {}

    most_frequent_state = max(empirical_states_dict.items(), key=lambda x: x[1])[0]

    bits_to_flip = [i for i, bit in enumerate(most_frequent_state) if bit == 1]

    emp_gauged_states_dict = {}

    for state, prob in empirical_states_dict.items():
        new_state = list(state)

        for bit_pos in bits_to_flip:
            new_state[bit_pos] = 1 - new_state[bit_pos]

        emp_gauged_states_dict[tuple(new_state)] = prob

    return emp_gauged_states_dict


# _____________________________________________________________________________________________________


# To be used to find the best permutation of the states that minimizes the KL divergence with the HFM model.
# The same permutation should be valid for all g values, therefore it is not necessary to recalculate the gauge for different g values. See ```print_minimum_kl_in_g_range``` below.
def find_minimum_kl_brute_force(
    good_gauged_probs,
    g=np.log(2),
    return_gauged_states_dict=True,
    print_permutation_steps=float("inf"),
):  # for return_minimum_kl
    """
    Brute-force search for the permutation of state columns that minimizes the KL divergence
    between the permuted empirical distribution and the HFM model with parameter g.

    Args:
        good_gauged_probs (dict): Dictionary mapping state tuples to probabilities.
        g (float, optional): HFM model parameter. Defaults to np.log(2).
        return_gauged_states_dict (bool, optional): If True, returns the best permutation and state dict.

    Returns:
        minimum_kl (float): The minimum KL divergence found.
        best_permutation (tuple or None): The permutation that yields the minimum KL (if requested).
        best_state_dict (dict or None): The permuted state dictionary (if requested).
    """
    states_matrix = np.array(list(good_gauged_probs.keys()))
    state_len = states_matrix.shape[1]
    permutations_list = list(permutations(range(state_len)))

    total_permutations = len(permutations_list)

    minimum_kl = float("inf")
    best_permutation = None
    gauged_states_dict = None
    i = 0

    for perm in permutations_list:
        i += 1

        states_matrix_copy = np.empty(states_matrix.shape)
        states_matrix_copy[:, :] = states_matrix[:, list(perm)]

        permutated_state_dict = dict(
            (tuple(row), prob)
            for row, prob in zip(states_matrix_copy, good_gauged_probs.values())
        )

        temporary_kl = calculate_kl_divergence_with_HFM(permutated_state_dict, g=g)
        if temporary_kl < minimum_kl:
            minimum_kl = temporary_kl
            best_permutation = perm
            gauged_states_dict = permutated_state_dict

        if i % print_permutation_steps == 0:
            print(
                f"Processed {i} permutations, current minimum KL: {minimum_kl}, best permutation: {best_permutation}"
            )

    print(
        f"Total permutations processed: {total_permutations}, Minimum KL: {minimum_kl}, Best permutation: {best_permutation}"
    )

    return (minimum_kl, gauged_states_dict) if return_gauged_states_dict else minimum_kl


# RENAME VARIABLES
def find_minimum_kl_simulated_annealing(
    good_gauged_probs,
    g=np.log(2),
    return_gauged_states_dict=True,
    initial_temp=10.0,
    cooling_rate=0.95,
    n_iterations=1000,
    verbose=False,
):
    """
    Uses simulated annealing to find a permutation of state columns that minimizes
    the KL divergence with the HFM model.
    """
    states_matrix = np.array(list(good_gauged_probs.keys()))
    state_len = states_matrix.shape[1]

    # Start with identity permutation
    current_perm = list(range(state_len))

    # Create initial state dictionary
    states_matrix_copy = np.empty(states_matrix.shape)
    states_matrix_copy[:, :] = states_matrix[:, current_perm]
    current_state_dict = dict(
        (tuple(row), prob)
        for row, prob in zip(states_matrix_copy, good_gauged_probs.values())
    )

    # Calculate initial KL divergence
    current_kl = calculate_kl_divergence_with_HFM(current_state_dict, g=g)

    # Keep track of best solution
    best_perm = current_perm.copy()
    best_kl = current_kl
    best_state_dict = current_state_dict.copy()

    # Temperature schedule
    temp = initial_temp

    # Main simulated annealing loop
    for i in range(n_iterations):
        # Use combinations from itertools to select positions to swap
        swap_indices = random.choice(list(combinations(range(state_len), 2)))

        # Create a new candidate permutation by swapping positions
        candidate_perm = current_perm.copy()
        candidate_perm[swap_indices[0]], candidate_perm[swap_indices[1]] = (
            candidate_perm[swap_indices[1]],
            candidate_perm[swap_indices[0]],
        )

        # Calculate KL for the candidate permutation
        states_matrix_copy[:, :] = states_matrix[:, candidate_perm]
        candidate_state_dict = dict(
            (tuple(row), prob)
            for row, prob in zip(states_matrix_copy, good_gauged_probs.values())
        )
        candidate_kl = calculate_kl_divergence_with_HFM(candidate_state_dict, g=g)

        # Metropolis acceptance criterion
        delta_kl = candidate_kl - current_kl
        if delta_kl < 0 or random.random() < math.exp(-delta_kl / temp):
            current_perm = candidate_perm
            current_kl = candidate_kl
            current_state_dict = candidate_state_dict

            # Update best solution if applicable
            if current_kl < best_kl:
                best_perm = current_perm.copy()
                best_kl = current_kl
                best_state_dict = current_state_dict.copy()

                if verbose:
                    print(
                        f"Iteration {i + 1}, New best KL: {best_kl:.6f}, Temperature: {temp:.6f}"
                    )

        # Reduce temperature
        temp *= cooling_rate

        # Periodic progress report
        if verbose and (i + 1) % 100 == 0:
            print(
                f"Iteration {i + 1}, Current KL: {current_kl:.6f}, Best KL: {best_kl:.6f}"
            )

    if verbose:
        print(f"Final KL: {best_kl:.6f}, Best permutation: {tuple(best_perm)}")

    return (best_kl, best_state_dict) if return_gauged_states_dict else best_kl





def get_empirical_gauged_states_dict(
        model,
        data_loader,
        brute_force = False,
        verbose = False,
        threshold_for_binarization = 0.5
    ):
    
    empirical_states_dict = get_empirical_states_dict(
        model, data_loader, verbose=verbose, threshold_for_binarization=threshold_for_binarization
    )
    flipped_states_dict = flip_gauge_bits(empirical_states_dict)

    if brute_force:
        minimum_kl, gauged_states_dict = find_minimum_kl_brute_force(
            flipped_states_dict,
            g=np.log(2),
            return_gauged_states_dict=True,
            verbose=verbose,
        )
    else:
        minimum_kl, gauged_states_dict = find_minimum_kl_simulated_annealing(
            flipped_states_dict,
            g=np.log(2),
            return_gauged_states_dict=True,
            verbose=verbose,
        )

    return gauged_states_dict




# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––



def get_ms_mean(gauged_states):
    """
    Calculate the weighted mean of m_s, where weights are the frequencies (probabilities) in gauged_states.
    """
    ms_values = [get_m_s(state, False) for state in gauged_states.keys()]
    weights = [gauged_states[state] for state in gauged_states.keys()]
    return np.average(ms_values, weights=weights)




def get_optimal_g(gauged_states, plot_graph=False, verbose=False):
    """
    Find the optimal g value that maximizes the mean of expected values.
    """
    ms_mean = get_ms_mean(gauged_states)
    latent_dim = len(next(iter(gauged_states)))

    gs = np.linspace(-3, 3, 1000)

    y = []
    for g in gs:
        y.append(-np.log(calculate_Z_theoretical(latent_dim, g)))

    ms_average = np.gradient(y, gs)

    nearest_position = np.argmin(np.abs(ms_average - ms_mean))
    nearest_g = gs[nearest_position]
    nearest_value = ms_average[nearest_position]

    if plot_graph:
        plt.plot(gs, ms_average, label="m_s average")
        plt.axhline(ms_mean, color="r", linestyle="--", label="ms_mean")
        plt.axvline(
            nearest_g, color="g", linestyle="--", label=f"optimal g = {nearest_g:.3f}"
        )
        plt.legend()
        plt.xlabel("g")
        plt.ylabel("Expected Value")
        plt.title("m_s average vs m_s mean")
        plt.show()

    if verbose:
        print(f"m_s mean: {ms_mean}")
        print(f"Optimal g: {nearest_g}, with expected value: {nearest_value}")

    return nearest_g




def get_KL_with_HFM_with_optimal_g(model, data_loader, return_g=False, threshold_for_binarization=0.5):   # IMPORTED IN DEPTH_ANALYSIS
    """
    Calculate the KL divergence between the empirical states and the HFM with the optimal g.
    """
    gauged_states_dict = get_empirical_gauged_states_dict(model, data_loader, threshold_for_binarization=threshold_for_binarization)
    optimal_g = get_optimal_g(gauged_states_dict)
    kl_divergence = calculate_kl_divergence_with_HFM(gauged_states_dict, optimal_g)

    return kl_divergence, optimal_g if return_g else kl_divergence


#–––––––––––––––––––––––––––––––––––––––––PLOTS–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––



def plot_KLs_vs_hidden_layers(KLs, gs, dataset_name):                                  # IMPORTED IN DEPTH_ANALYSIS
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
    plt.show()


def datasets_dicts_comparison(KLs_dict):                                            # IMPORTED IN DEPTH_ANALYSIS
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
    plt.show()
