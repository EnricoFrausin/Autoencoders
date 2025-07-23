import torch
from collections import defaultdict
import matplotlib.pyplot as plt



def get_binary_latent_frequencies(model, dataloader):
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
            binary_states = (latent_vectors >= 0.5).int()
            
            # Convert each binary vector to tuple (hashable for dictionary keys)
            for i in range(binary_states.size(0)):
                state_tuple = tuple(binary_states[i].cpu().numpy())
                state_counts[state_tuple] += 1
                total_samples += 1

    # Normalize frequencies by total_samples
    frequency_dict = {k: v / total_samples for k, v in state_counts.items()}

    
    print(f"Total samples processed: {total_samples}")
    print(f"Number of unique binary states found: {len(frequency_dict)}")
    #print(f"Theoretical maximum states for {model.latent_dim}-dim latent: {2**model.#latent_dim}")
    
    return frequency_dict



def analyze_binary_frequencies(frequency_dict, top_k=10):
    """
    Analyze and display the most frequent binary states.
    
    Args:
        frequency_dict: Dictionary from get_binary_latent_frequencies
        top_k: Number of top states to display
    """
    import matplotlib.pyplot as plt
    
    # Sort by frequency (descending)
    sorted_states = sorted(frequency_dict.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {top_k} most frequent binary states:")
    print("-" * 50)
    for i, (state, count) in enumerate(sorted_states[:top_k]):
        percentage = (count / sum(frequency_dict.values())) * 100
        state_str = ''.join(map(str, state))
       # print(f"{i+1:2d}. {state_str} -> {count:5d} samples ({percentage:5.2f}%)")
    
    # Plot frequency distribution
    frequencies = [count for _, count in sorted_states]
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(frequencies)), frequencies)
    plt.xlabel('Binary State (sorted by frequency)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Binary States')
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    plt.bar(range(min(top_k, len(frequencies))), frequencies[:top_k])
    plt.xlabel('Top Binary States')
    plt.ylabel('Frequency')
    plt.title(f'Top {top_k} Most Frequent States')
    
    plt.tight_layout()
    plt.show()
    
    return sorted_states
