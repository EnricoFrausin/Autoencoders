import numpy as np
from itertools import permutations, combinations
import os
import matplotlib.pyplot as plt
import random
import math

from AE.utils import calculate_kl_divergence_with_HFM
from AE.utils import get_emp_states_dict
from AE.utils import calculate_Z_theoretical
from AE.utils import get_m_s




#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––



def flip_gauge_bits(emp_states_dict):  # for return_minimum_kl
    """
    Flip specific bits in all states based on the activated bits in the most frequent state.

    Args:
        emp_states_dict (dict): Dictionary mapping state tuples to their empirical probabilities.
                                From get_emp_states_dict.

    Returns:
        dict: A new dictionary with the same probabilities but flipped states according to the rule.
    """

    most_frequent_state = max(emp_states_dict.items(), key=lambda x: x[1])[0]

    bits_to_flip = [i for i, bit in enumerate(most_frequent_state) if bit == 1]

    emp_gauged_states_dict = {}

    for state, prob in emp_states_dict.items():
        new_state = list(state)

        for bit_pos in bits_to_flip:
            new_state[bit_pos] = 1 - new_state[bit_pos]

        emp_gauged_states_dict[tuple(new_state)] = prob

    return emp_gauged_states_dict


# _____________________________________________________________________________________________________


# To be used to find the best permutation of the states that minimizes the KL divergence with the HFM model.
# The same permutation should be valid for all g values, therefore it is not necessary to recalculate the gauge for different g values. See ```print_minimum_kl_in_g_range``` below.
def find_minimum_kl_brute_force(
    emp_states_dict_flipgauged,
    g=np.log(2),
    return_gauged_states_dict=True,
    print_permutation_steps=float("inf"),
):  # for return_minimum_kl
    """
    Brute-force search for the permutation of state columns that minimizes the KL divergence
    between the permuted empirical distribution and the HFM model with parameter g.

    Args:
        emp_states_dict_flipgauged (dict): Dictionary mapping state tuples to probabilities.
        g (float, optional): HFM model parameter. Defaults to np.log(2).
        return_gauged_states_dict (bool, optional): If True, returns the best permutation and state dict.

    Returns:
        minimum_kl (float): The minimum KL divergence found.
        best_permutation (tuple or None): The permutation that yields the minimum KL (if requested).
        best_state_dict (dict or None): The permuted state dictionary (if requested).
    """
    emp_states_matrix = np.array(list(emp_states_dict_flipgauged.keys()))
    state_len = emp_states_matrix.shape[1]
    permutations_list = list(permutations(range(state_len)))

    total_permutations = len(permutations_list)

    minimum_kl = float("inf")
    best_permutation = None
    emp_states_dict_gauged = None
    i = 0

    for perm in permutations_list:
        i += 1

        states_matrix_copy = np.empty(emp_states_matrix.shape)
        states_matrix_copy[:, :] = emp_states_matrix[:, list(perm)]

        permutated_state_dict = dict(
            (tuple(row), prob)
            for row, prob in zip(states_matrix_copy, emp_states_dict_flipgauged.values())
        )

        temporary_kl = calculate_kl_divergence_with_HFM(permutated_state_dict, g=g)
        if temporary_kl < minimum_kl:
            minimum_kl = temporary_kl
            best_permutation = perm
            emp_states_dict_gauged = permutated_state_dict

        if i % print_permutation_steps == 0:
            print(
                f"Processed {i} permutations, current minimum KL: {minimum_kl}, best permutation: {best_permutation}"
            )

    print(
        f"Total permutations processed: {total_permutations}, Minimum KL: {minimum_kl}, Best permutation: {best_permutation}"
    )

    return (minimum_kl, emp_states_dict_gauged) if return_gauged_states_dict else minimum_kl


# RENAME VARIABLES
def find_minimum_kl_simulated_annealing(
    emp_states_dict_flipgauged,
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
    emp_states_matrix = np.array(list(emp_states_dict_flipgauged.keys()))
    state_len = emp_states_matrix.shape[1]

    # Start with identity permutation
    current_perm = list(range(state_len))

    # Create initial state dictionary
    states_matrix_copy = np.empty(emp_states_matrix.shape)
    states_matrix_copy[:, :] = emp_states_matrix[:, current_perm]
    current_state_dict = dict(
        (tuple(row), prob)
        for row, prob in zip(states_matrix_copy, emp_states_dict_flipgauged.values())
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
        states_matrix_copy[:, :] = emp_states_matrix[:, candidate_perm]
        candidate_state_dict = dict(
            (tuple(row), prob)
            for row, prob in zip(states_matrix_copy, emp_states_dict_flipgauged.values())
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





def get_emp_states_dict_gauged(
        model,
        data_loader,
        brute_force = False,
        verbose = False,
        threshold_for_binarization = 0.5
    ):
    
    emp_states_dict = get_emp_states_dict(
        model, data_loader, verbose=verbose, threshold_for_binarization=threshold_for_binarization
    )
    flipped_states_dict = flip_gauge_bits(emp_states_dict)

    if brute_force:
        minimum_kl, emp_states_dict_gauged = find_minimum_kl_brute_force(
            flipped_states_dict,
            g=np.log(2),
            return_gauged_states_dict=True,
            verbose=verbose,
        )
    else:
        minimum_kl, emp_states_dict_gauged = find_minimum_kl_simulated_annealing(
            flipped_states_dict,
            g=np.log(2),
            return_gauged_states_dict=True,
            verbose=verbose,
        )

    return emp_states_dict_gauged




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




def get_KL_with_HFM_with_optimal_g(model, data_loader, return_g=False, threshold_for_binarization=0.5):   # EXPORTED TO DEPTH_ANALYSIS
    """
    Calculate the KL divergence between the empirical states and the HFM with the optimal g.
    """
    emp_states_dict_gauged = get_emp_states_dict_gauged(model, data_loader, threshold_for_binarization=threshold_for_binarization)
    optimal_g = get_optimal_g(emp_states_dict_gauged)
    kl_divergence = calculate_kl_divergence_with_HFM(emp_states_dict_gauged, optimal_g)

    return kl_divergence, optimal_g if return_g else kl_divergence




def get_bottleneck_neurons_frequencies(model, dataloader, threshold_for_binarization=0.5):
    """
    Calculates the activation frequencies of bottleneck neurons in an autoencoder model.
    This function computes how frequently each neuron in the bottleneck layer is activated
    across the dataset provided by the dataloader. The activations are binarized using the
    specified threshold before calculating the frequencies.
    Args:
        model (torch.nn.Module): The autoencoder model containing the bottleneck layer.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the input data.
        threshold_for_binarization (float, optional): Threshold to binarize neuron activations.
            Defaults to 0.5.
    Returns:
        dict: A dictionary mapping each bottleneck neuron index to its activation frequency
        (i.e., the proportion of samples for which the neuron is active).
    """
    
    emp_states_dict = get_emp_states_dict(model, dataloader, threshold_for_binarization)

    return calculate_neurons_activation_frequencies(emp_states_dict)




def calculate_neurons_activation_frequencies(emp_states_dict):
    """
    Given a dictionary where keys are binary tuples (or lists) and values are frequencies,
    returns a vector with the weighted sum (activation frequency) for each neuron/dimension.

    Args:
        emp_states_dict (dict): {(0,1,1,...): freq, ...}

    Returns:
        np.ndarray: activation frequency for each dimension
    """
    # Convert keys to numpy array
    states = np.array(list(emp_states_dict.keys()))
    freqs = np.array(list(emp_states_dict.values()))
    # Weighted sum along each column (dimension)
    activation_freqs = np.average(states, axis=0, weights=freqs)
    return activation_freqs
