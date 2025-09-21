import numpy as np
from itertools import permutations, combinations
import os
import matplotlib.pyplot as plt
import random
import math
from torch.nn import Module
from torch.utils.data import DataLoader
import torch

from AE.utils import calc_hfm_kld
from AE.utils import compute_emp_states_dict
from AE.utils import calc_Z_theoretical
from AE.utils import calc_ms

from AE.models import AE_0


# –––––––––––––––––––––––––––––––––––– EXPORTED TO DEPTH_ANALYSIS –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

def compute_dataset_klds_gs_dict_with_optimal_threshold_(dataset, data_loader, device, model_kwargs, model_path_kwargs, binarize_threshold_range, num_hidden_layers_range, dataset_klds_dict = None, dataset_gs_dict = None, verbose=True):
    """
    Computes and stores the KLDs and gs values for a given dataset by searching for the optimal binarization threshold
    that minimizes a custom energy function.

    Args:
        dataset (str): Name of the dataset (e.g., 'MNIST', 'EMNIST', '2MNIST').
        data_loader (DataLoader): DataLoader for the dataset to evaluate.
        model_kwargs (dict): Keyword arguments for constructing the model (passed to AE_0).
        model_path_kwargs (dict): Keyword arguments for constructing the model path.
        binarize_threshold_range (iterable): Range of binarization thresholds to search over.
        num_hidden_layers_range (iterable): Range of hidden layer counts to evaluate.
        dataset_klds_dict (dict, optional): Dictionary to store KLDs for each dataset. If None, a new one is created.
        dataset_gs_dict (dict, optional): Dictionary to store gs values for each dataset. If None, a new one is created.
        verbose (bool, optional): If True, prints the best threshold and energy found.

    Returns:
        tuple: (dataset_klds_dict, dataset_gs_dict) with the results for the given dataset and optimal threshold.
    """

    if dataset_klds_dict is None:
        dataset_klds_dict = {
            '2MNIST': [],
            'MNIST': [],
            'EMNIST': []}
    if dataset_gs_dict is None:
        dataset_gs_dict = {
            '2MNIST': [],
            'MNIST': [],
            'EMNIST': []}

    dataset_klds_dict[dataset] = []
    dataset_gs_dict[dataset] = []

    memoize_klds_dict = {}
    memoize_gs_dict = {}

    klds_sign_changes_lst, gs_distances_lst = compute_energy_addends_lst(
        binarize_threshold_range=binarize_threshold_range, 
        dataset=dataset,
        device=device, 
        data_loader=data_loader, 
        model_kwargs=model_kwargs, 
        model_path_kwargs=model_path_kwargs, 
        num_hidden_layers_range=num_hidden_layers_range, 
        memoize_klds_dict=memoize_klds_dict, 
        memoize_gs_dict=memoize_gs_dict
    )

    best_binarize_thresold, best_energy = optimize_binarize_threshold_through_energy(
        klds_sign_changes_lst, 
        gs_distances_lst, 
        binarize_threshold_range
    )

    dataset_klds_dict[dataset] = memoize_klds_dict[str(best_binarize_thresold)]
    dataset_gs_dict[dataset] = memoize_gs_dict[str(best_binarize_thresold)]

    if verbose:
        print(f"best_binarize_thresold: ", best_binarize_thresold)
        print(f"best_energy: ", best_energy)

    return dataset_klds_dict, dataset_gs_dict



# ––––––––––––––––––––––––––––––––––– NEEDED FOR compute_dataset_klds_gs_dict_with_optimal_threshold_ ––––––––––––––––––––––––––––––––––––––



def optimize_binarize_threshold_through_energy(klds_sign_changes_lst, gs_distances_lst, binarize_threshold_range):
    """
    Finds the optimal binarization threshold that minimizes a custom energy function.

    The energy function is defined as the sum of the normalized number of KLD sign changes
    and the normalized sum of distances of gs values from log(2), for each threshold.

    Args:
        klds_sign_changes_lst (list or np.ndarray): List of normalized KLD sign change counts for each threshold.
        gs_distances_lst (list or np.ndarray): List of normalized gs distances from log(2) for each threshold.
        binarize_threshold_range (iterable): Range of binarization thresholds considered.

    Returns:
        tuple: (best_binarize_threshold, best_energy)
            - best_binarize_threshold (float): The threshold value that minimizes the energy.
            - best_energy (float): The minimum energy value found.
    """
    best_energy = float('inf')
    best_binarize_thresold = None

    for i, binarize_threshold in enumerate(binarize_threshold_range):
        energy = (
            1 * klds_sign_changes_lst[i] 
            + 1 * gs_distances_lst[i]
        )
        if energy < best_energy:
            best_energy = energy
            best_binarize_thresold = binarize_threshold

    return best_binarize_thresold, best_energy



def compute_energy_addends_lst(binarize_threshold_range, dataset, data_loader, device, model_kwargs, model_path_kwargs, num_hidden_layers_range, memoize_klds_dict, memoize_gs_dict):
    """
    Computes the normalized energy addends for a range of binarization thresholds.

    For each threshold in `binarize_threshold_range`, this function:
      - Computes the list of KLDs and gs values across different hidden layer counts.
      - Calculates the number of sign changes in the KLDs list.
      - Calculates the sum of absolute differences between each gs value and log(2).
      - Normalizes both lists by their respective maxima.

    Args:
        binarize_threshold_range (iterable): Range of binarization thresholds to evaluate.
        dataset (str): Name of the dataset.
        data_loader (DataLoader): DataLoader for the dataset.
        model_kwargs (dict): Arguments for model construction.
        model_path_kwargs (dict): Arguments for model path construction.
        num_hidden_layers_range (iterable): Range of hidden layer counts to evaluate.
        memoize_klds_dict (dict): Dictionary to store computed KLDs for each threshold.
        memoize_gs_dict (dict): Dictionary to store computed gs values for each threshold.

    Returns:
        tuple:
            - klds_sign_changes_lst (np.ndarray): Normalized list of KLD sign change counts for each threshold.
            - gs_distances_lst (np.ndarray): Normalized list of gs distances from log(2) for each threshold.
    """
    klds_sign_changes_lst = []
    gs_distances_lst = []

    for binarize_threshold in binarize_threshold_range:

        klds_lst, gs_lst = compute_klds_gs_lst_with_fixed_threshold(
            data_loader=data_loader, 
            device=device, 
            model_kwargs=model_kwargs, 
            model_path_kwargs=model_path_kwargs, 
            binarize_threshold=binarize_threshold,
            num_hidden_layers_range=num_hidden_layers_range, 
            memoize_klds_dict=memoize_klds_dict, 
            memoize_gs_dict=memoize_gs_dict)

        klds_sign_changes_lst.append( count_sign_changes(klds_lst) )
        gs_distances_lst.append( sum(abs(a - np.log(2)) for a in gs_lst) )

    klds_sign_changes_lst = np.array(klds_sign_changes_lst) / max(klds_sign_changes_lst)
    gs_distances_lst = np.array(gs_distances_lst) / max(gs_distances_lst)

    return klds_sign_changes_lst, gs_distances_lst




def compute_klds_gs_lst_with_fixed_threshold(data_loader, model_kwargs, device, model_path_kwargs, binarize_threshold, num_hidden_layers_range, memoize_klds_dict, memoize_gs_dict):
    """
    Computes lists of KLD and gs values for a fixed binarization threshold across a range of hidden layer counts.

    For each number of hidden layers in `num_hidden_layers_range`, this function:
      - Loads the corresponding model using the provided `model_kwargs` and `model_path_kwargs`.
      - Computes the Kullback-Leibler divergence (KLD) and gs value using `calc_hfm_kld_with_optimal_g`.
      - Appends the results to lists.

    Optionally memoizes the results for the given threshold in the provided dictionaries.

    Args:
        data_loader (DataLoader): DataLoader for the dataset to evaluate.
        model_kwargs (dict): Arguments for constructing the model (passed to AE_0).
        model_path_kwargs (dict): Arguments for constructing the model path.
        binarize_threshold (float): The binarization threshold to use.
        num_hidden_layers_range (iterable): Range of hidden layer counts to evaluate.
        memoize_klds_dict (dict): Dictionary to store computed KLDs for each threshold.
        memoize_gs_dict (dict): Dictionary to store computed gs values for each threshold.

    Returns:
        tuple: (klds_lst, gs_lst)
            - klds_lst (list): List of KLD values for each hidden layer count.
            - gs_lst (list): List of gs values for each hidden layer count.
    """
    from AE.depth_utils import calc_hfm_kld_with_optimal_g

    klds_lst = []
    gs_lst = []

    for num_hidden_layers in num_hidden_layers_range:

        my_model = AE_0(
            **model_kwargs,
            hidden_layers=num_hidden_layers
        ).to(device)
        model_path = f"../models/{model_path_kwargs['output_activation_encoder']}/{model_path_kwargs['initialization']}/{model_path_kwargs['train_type']}/{model_path_kwargs['latent_dim']}/{model_path_kwargs['dataset']}/lr{model_path_kwargs['learning_rate']}_dr{model_path_kwargs['decrease_rate']}_bias{model_path_kwargs['bias']}_{num_hidden_layers}hl_{model_path_kwargs['train_num']}.pth"
        my_model.load_state_dict(torch.load(model_path, map_location=device))

        current_kld, current_g = calc_hfm_kld_with_optimal_g(my_model, data_loader, return_g=True, binarize_threshold=binarize_threshold)
        
        gs_lst.append(current_g)
        klds_lst.append(current_kld)

    if memoize_klds_dict is not None:
        memoize_klds_dict[str(binarize_threshold)] = klds_lst
    if memoize_gs_dict is not None:
        memoize_gs_dict[str(binarize_threshold)] = gs_lst

    return klds_lst, gs_lst



def count_sign_changes(values):
    """
    Counts the number of sign changes in the first differences of a sequence.
    Args:
        values (list or np.ndarray): Input sequence of numbers.
    Returns:
        int: The computed 'sign_changes' (number of sign changes).
    """
    diff_0 = 0
    sign_changes = 0
    for a, b in zip(values[:-1], values[1:]):
        diff_1 = b - a
        if diff_0 == 0:
            diff_0 = diff_1
            continue
        if np.sign(diff_1) != np.sign(diff_0):
            sign_changes += 1*abs(diff_1)
        diff_0 = diff_1
    return sign_changes




# –––––––––––––––––––––––––––––––––– EXPORTED TO DETPH_ANALYSIS ––––––––––––––––––––––––––––––––––––––



def compute_bottleneck_neurons_activ_freq(
        model: Module,
        dataloader: DataLoader,
        binarize_threshold = 0.5,
        flip_gauge = False
):          
    """
    Calculates the activation frequencies of bottleneck neurons in an autoencoder model.
    This function computes how frequently each neuron in the bottleneck layer is activated
    across the dataset provided by the dataloader. The activations are binarized using the
    specified threshold before calculating the frequencies.
    Args:
        model (torch.nn.Module): The autoencoder model containing the bottleneck layer.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the input data.
        binarize_threshold (float, optional): Threshold to binarize neuron activations.
            Defaults to 0.5.
        flip_gauge (bool, optional): If True, flips the bits of emp_states such that the most frequent state is the all zero entries state.
            Defaults to false.
    Returns:
        np.ndarray: A numpy array with activation frequency as entries.
        (i.e., the proportion of samples for which the neuron is active).
    """
    
    emp_states_dict = compute_emp_states_dict(model, dataloader, binarize_threshold)

    if flip_gauge:
        emp_states_dict = flip_gauge_bits(emp_states_dict)
    
    bottleneck_activ_freq = calc_neurons_activ_freq(emp_states_dict)

    return bottleneck_activ_freq



def compute_bottleneck_neurons_activ_freq_gauged(
        model: Module,
        dataloader: DataLoader,
        binarize_threshold = 0.5,
):          
    """
    Calculates the activation frequencies of bottleneck neurons in an autoencoder model, after finding the correct gauge.
    This function computes how frequently each neuron in the bottleneck layer is activated
    across the dataset provided by the dataloader. The activations are binarized using the
    specified threshold before calculating the frequencies.
    Args:
        model (torch.nn.Module): The autoencoder model containing the bottleneck layer.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the input data.
        binarize_threshold (float, optional): Threshold to binarize neuron activations.
            Defaults to 0.5.
        flip_gauge (bool, optional): If True, flips the bits of emp_states such that the most frequent state is the all zero entries state.
            Defaults to false.
    Returns:
        np.ndarray: A numpy array with activation frequency as entries.
        (i.e., the proportion of samples for which the neuron is active).
    """
    
    emp_states_dict_gauged = compute_emp_states_dict_gauged(model, dataloader, binarize_threshold)
    
    bottleneck_activ_freq = calc_neurons_activ_freq(emp_states_dict_gauged)

    return bottleneck_activ_freq



def calc_hfm_kld_with_optimal_g(            # used in compute_klds_gs_lst_with_fixed_threshold AND EXPORTED TO DEPTH_ANALYSIS
        model: Module,
        data_loader: DataLoader,
        return_g = False,
        binarize_threshold = 0.5):   
    """
    Calculates the KL divergence between the empirical latent state distribution (from the model and data_loader)
    and the HFM model with its optimal parameter g (chosen to best match the empirical mean).

    Args:
        model (torch.nn.Module): The trained autoencoder model.
        data_loader (torch.utils.data.DataLoader): DataLoader providing input data.
        return_g (bool, optional): If True, also returns the optimal g value. Defaults to False.
        binarize_threshold (float, optional): Threshold for binarizing latent activations. Defaults to 0.5.

    Returns:
        float: KL divergence between empirical states and HFM with optimal g.
        float (optional): The optimal g value (if return_g is True).
    """
    emp_states_dict_gauged = compute_emp_states_dict_gauged(model, data_loader, binarize_threshold=binarize_threshold)
    optimal_g = calc_optimal_g(emp_states_dict_gauged)
    kl_div = calc_hfm_kld(emp_states_dict_gauged, optimal_g)

    return kl_div, optimal_g if return_g else kl_div



#–––––––––––––––––––––––––––––––––––––––– USED IN THE FUNCTIONS DEFINED ABOVE - 1 ––––––––––––––––––––––––––––––––––––––––––––––––––



def calc_neurons_activ_freq(emp_states_dict):       # USED IN compute_bottleneck_neurons_activ_freq
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



def compute_emp_states_dict_gauged(                 # USED IN calc_hfm_kld_with_optimal_g AND EXPORTED TO DEPTH.UTILS
        model: Module,
        data_loader: DataLoader,
        binarize_threshold = 0.5,
        flip_gauge = True,
        brute_force = False,
        return_perm = False,
        verbose = False
    ):
    """
    Computes the empirical latent state distribution from a model and dataloader, applies gauge flipping
    (bit inversion based on the most frequent state), and finds the permutation of latent dimensions
    that minimizes the KL divergence with the HFM model. The permutation is found using either brute-force
    search or simulated annealing.

    Args:
        model (torch.nn.Module): The trained autoencoder model.
        data_loader (torch.utils.data.DataLoader): DataLoader providing input data.
        binarize_threshold (float, optional): Threshold for binarizing latent activations. Defaults to 0.5.
        flip_gauge (bool, optional): If True, flips the bits of emp_states such that the most frequent state is the all zero entries state.
        brute_force (bool, optional): If True, uses brute-force search for optimal permutation; otherwise uses simulated annealing. Defaults to False.
        return_perm (bool, optional): If True, returns both the gauged state dictionary and the optimal permutation. Defaults to False.
        verbose (bool, optional): If True, prints progress information during permutation search. Defaults to False.

    Returns:
        dict: Empirical state dictionary with columns permuted and gauge-flipped to minimize KL divergence with the HFM model.
        tuple (optional): The optimal permutation of latent dimensions (if return_perm is True).
    """

    emp_states_dict = compute_emp_states_dict(
        model, data_loader, verbose=verbose, binarize_threshold=binarize_threshold
    )

    if flip_gauge:
        emp_states_dict = flip_gauge_bits(emp_states_dict)

    if brute_force:
        gauge_perm = compute_perm_minimizing_hfm_kld_brute_force(emp_states_dict, verbose=verbose)
    else:
        gauge_perm = compute_perm_minimizing_hfm_kld_simul_anneal(emp_states_dict, verbose=verbose)

    emp_states_dict_gauged = {tuple(k[i] for i in gauge_perm): v for k, v in emp_states_dict.items()}

    if return_perm:
        return emp_states_dict_gauged, gauge_perm
    else:
        return emp_states_dict_gauged



def calc_optimal_g(                                 # USED IN calc_hfm_kld_with_optimal_g
        emp_states_dict_gauged,
        plot_graph = False,
        verbose = False
):
    """
    Finds the optimal value of the parameter 'g' for the HFM such that
    the theoretical mean of the expected value (m_s average) matches the empirical mean (m_s mean)
    computed from the provided gauged states.

    The function computes the empirical mean of m_s from the gauged states, then searches for the
    value of 'g' where the gradient of the log partition function (theoretical m_s average) is
    closest to the empirical mean. Optionally, it can plot the comparison and print verbose output.

    Args:
        emp_states_dict_gauged (dict): Dictionary mapping state tuples to their empirical probabilities.
        plot_graph (bool, optional): If True, plots the theoretical m_s average, empirical mean,
            and the optimal g. Defaults to False.
        verbose (bool, optional): If True, prints the empirical mean, optimal g, and corresponding
            theoretical value. Defaults to False.

    Returns:
        float: The optimal value of g that best matches the empirical m_s mean.
    """

    ms_mean = calc_ms_mean(emp_states_dict_gauged)              # Empirical value to compare with theoretical ms_average values
    latent_dim = len(next(iter(emp_states_dict_gauged)))

    gs = np.linspace(-3, 3, 1000)                               # g domain to calculate -log(Z(g))

    y = []
    for g in gs:
        y.append(-np.log(calc_Z_theoretical(latent_dim, g)))    
    ms_average = np.gradient(y, gs)                             # Theoretical values for comparison with ms_mean. Numerical gradients of y over g (vector).

    nearest_position = np.argmin(np.abs(ms_average - ms_mean))
    nearest_g = gs[nearest_position]
    nearest_value = ms_average[nearest_position]

    # --------------------------------

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



#–––––––––––––––––––––––––––––––––––––––– USED IN THE FUNCTIONS DEFINED ABOVE - 2 ––––––––––––––––––––––––––––––––––––––––––––––––––



def flip_gauge_bits(emp_states_dict):                       # USED IN compute_emp_states_dict_gauged
    """
    Flip specific bits in all states based on the activated bits in the most frequent state.

    Args:
        emp_states_dict (dict): Dictionary mapping state tuples to their empirical probabilities.
                                From compute_emp_states_dict.

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



# To be used to compute the best permutation of the states that minimizes the KL divergence with the HFM model.
# The same permutation should be valid for all g values, therefore it is not necessary to recalculate the gauge for different g values. See ```print_minimum_kl_in_g_range``` below.
def compute_perm_minimizing_hfm_kld_brute_force(
    emp_states_dict: dict,
    g = np.log(2),
    verbose = False,
):
    """
    Brute-force search for the permutation of state columns that minimizes the KL divergence
    between the permuted empirical distribution and the HFM model with parameter g.

    Args:
        emp_states_dict (dict): Dictionary mapping state tuples to probabilities.
        g (float, optional): HFM model parameter. Defaults to np.log(2). The best permutation should be independent from g.
        return_gauged_states_dict (bool, optional): If True, returns the best permutation and state dict.

    Returns:
        best_kl (float): The minimum KL divergence found.
        best_permutation (tuple or None): The permutation that yields the minimum KL (if requested).
        best_state_dict (dict or None): The permuted state dictionary (if requested).
    """


    state_len = len(list(emp_states_dict.keys())[0])

    permutations_list = list(permutations(range(state_len)))
    permutations_count = len(permutations_list)

    best_kl = float("inf")
    best_permutation = None

    i = 0
    for perm in permutations_list:
        i += 1

        permuted_states_dict = {tuple(k[i] for i in perm): v for k, v in emp_states_dict}

        current_kl = calc_hfm_kld(permuted_states_dict, g=g)
        if current_kl < best_kl:
            best_kl = current_kl
            best_permutation = perm

        if verbose:
            if i % 10000 == 0:
                print(
                    f"Processed {i}/{permutations_count} permutations, current minimum KL: {best_kl}, best permutation: {best_permutation}"
                )

    return best_permutation



def compute_perm_minimizing_hfm_kld_simul_anneal(
    emp_states_dict: dict,
    g = np.log(2),
    initial_temp = 10.0,
    cooling_rate = 0.95,
    n_iterations = 5000,
    return_perm = False,
    verbose = False
):
    """
    Uses simulated annealing to compute a permutation of state columns that minimizes
    the KL divergence between the empirical state distribution and the HFM model.

    Args:
        emp_states_dict (dict): Dictionary mapping state tuples to probabilities, with bits already gauge flipped.
        g (float, optional): HFM model parameter. Defaults to np.log(2). The best permutation should be independent from g.
        initial_temp (float, optional): Initial temperature for simulated annealing. Defaults to 10.0.
        cooling_rate (float, optional): Cooling rate for temperature decay. Defaults to 0.95.
        n_iterations (int, optional): Number of iterations for the annealing process. Defaults to 5000.
        verbose (bool, optional): If True, prints progress every 100 iterations. Defaults to False.

    Returns:
        list: The permutation of state columns that yields the lowest KL divergence found.
    """

    state_len = len(list(emp_states_dict.keys())[0])

    current_perm = list(range(state_len))                   # Identity permutation
    current_states_dict = emp_states_dict
    current_kl = calc_hfm_kld(current_states_dict, g=g)

    best_perm = current_perm                                # No need to .copy() because the variable is not modified
    best_kl = current_kl

    # Metropolis algorithm
    temp = initial_temp
    for i in range(n_iterations):

        swap_indices = random.choice(list(combinations(range(state_len), 2)))   

        candidate_perm = current_perm.copy()                                    # .copy() because the variable is modified
        candidate_perm[swap_indices[0]], candidate_perm[swap_indices[1]] = (    # swapped using tuple unpacking
            current_perm[swap_indices[1]], current_perm[swap_indices[0]]
        )   

        permuted_states_dict = {tuple(k[i] for i in candidate_perm): v for k, v in emp_states_dict.items()}
        candidate_kl = calc_hfm_kld(permuted_states_dict, g=g)

        delta_kl = candidate_kl - current_kl
        if delta_kl < 0 or random.random() < math.exp(-delta_kl / temp):

            current_perm = candidate_perm
            current_kl = candidate_kl

            if current_kl < best_kl:
                best_perm = current_perm.copy()
                best_kl = current_kl

        temp *= cooling_rate

        # Periodic progress report
        if verbose and (i + 1) % 100 == 0:
            print(
                f"Iteration {i + 1}, Current KL: {current_kl:.6f}, Best KL: {best_kl:.6f}"
            )

    return best_perm



def calc_ms_mean(emp_states_dict_gauged):
    """
    Calculates the weighted mean of the m_s statistic over all states in the provided dictionary.

    For each state, computes m_s using calc_ms(state, False), then averages these values weighted
    by the empirical probabilities (frequencies) from emp_states_dict_gauged.

    Args:
        emp_states_dict_gauged (dict): Dictionary mapping state tuples to their empirical probabilities.

    Returns:
        float: Weighted mean of m_s across all states.
    """
    ms_values = [calc_ms(state, False) for state in emp_states_dict_gauged.keys()]
    weights = [emp_states_dict_gauged[state] for state in emp_states_dict_gauged.keys()]
    return np.average(ms_values, weights=weights)


