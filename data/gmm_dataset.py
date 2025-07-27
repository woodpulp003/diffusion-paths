import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def generate_gmm_data(n_samples: int, n_components: int = 8, radius: float = 5.0, std: float = 0.2) -> np.ndarray:
    """
    Generate 2D data from a Gaussian Mixture Model where means are equally spaced on a circle.
    
    Args:
        n_samples: Number of samples to generate
        n_components: Number of Gaussian components (default: 8)
        radius: Radius of the circle where means are placed (default: 5.0)
        std: Standard deviation of each Gaussian component (default: 0.2)
    
    Returns:
        numpy array of shape (n_samples, 2) with generated data
    """
    # Generate means equally spaced on a circle
    angles = np.linspace(0, 2 * np.pi, n_components, endpoint=False)
    means = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Generate data from each component
    samples_per_component = n_samples // n_components
    data = []
    
    for i in range(n_components):
        # Generate samples from this component
        component_samples = np.random.normal(
            loc=means[i], 
            scale=std, 
            size=(samples_per_component, 2)
        )
        data.append(component_samples)
    
    # Handle any remaining samples
    remaining_samples = n_samples - (samples_per_component * n_components)
    if remaining_samples > 0:
        # Distribute remaining samples across components
        for i in range(remaining_samples):
            component_idx = i % n_components
            sample = np.random.normal(loc=means[component_idx], scale=std, size=(1, 2))
            data.append(sample)
    
    # Combine all samples
    data = np.vstack(data)
    
    # Shuffle the data
    np.random.shuffle(data)
    
    return data


def generate_complex_gmm_data(n_samples: int, n_components: int = 12, 
                             base_radius: float = 3.0, max_radius: float = 8.0,
                             min_std: float = 0.1, max_std: float = 0.4) -> np.ndarray:
    """
    Generate 2D data from a complex Gaussian Mixture Model with:
    - Varying variances (standard deviations)
    - Means at different radii (not uniform)
    - Non-uniform angle spacing
    - Different component densities
    
    Args:
        n_samples: Number of samples to generate
        n_components: Number of Gaussian components (default: 12)
        base_radius: Minimum radius for component placement (default: 3.0)
        max_radius: Maximum radius for component placement (default: 8.0)
        min_std: Minimum standard deviation (default: 0.1)
        max_std: Maximum standard deviation (default: 0.4)
    
    Returns:
        numpy array of shape (n_samples, 2) with generated data
    """
    # Generate non-uniform angles (some clusters closer together)
    angles = []
    for i in range(n_components):
        if i == 0:
            angles.append(0)
        elif i % 3 == 0:  # Every 3rd component gets a larger gap
            angles.append(angles[-1] + np.pi/3 + np.random.uniform(0, np.pi/6))
        else:  # Smaller gaps for other components
            angles.append(angles[-1] + np.pi/6 + np.random.uniform(-np.pi/12, np.pi/12))
    
    # Ensure angles are within [0, 2π]
    angles = np.array(angles) % (2 * np.pi)
    
    # Generate varying radii (not uniform)
    radii = np.random.uniform(base_radius, max_radius, n_components)
    # Make some components closer to center, others further out
    radii[::2] = np.random.uniform(base_radius, base_radius + 2, len(radii[::2]))
    radii[1::2] = np.random.uniform(max_radius - 2, max_radius, len(radii[1::2]))
    
    # Generate means with varying radii and angles
    means = np.column_stack([
        radii * np.cos(angles),
        radii * np.sin(angles)
    ])
    
    # Generate varying standard deviations
    stds = np.random.uniform(min_std, max_std, n_components)
    # Make some components more spread out, others more concentrated
    stds[::3] = np.random.uniform(max_std * 0.8, max_std, len(stds[::3]))  # More spread
    stds[1::3] = np.random.uniform(min_std, min_std * 1.5, len(stds[1::3]))  # More concentrated
    stds[2::3] = np.random.uniform(min_std * 1.2, max_std * 0.6, len(stds[2::3]))  # Medium
    
    # Generate varying sample counts per component (not uniform)
    # Some components will have more samples, others fewer
    component_weights = np.random.uniform(0.5, 2.0, n_components)
    component_weights = component_weights / np.sum(component_weights)  # Normalize
    
    samples_per_component = (component_weights * n_samples).astype(int)
    
    # Ensure we have exactly n_samples
    total_allocated = np.sum(samples_per_component)
    if total_allocated < n_samples:
        # Add remaining samples to components with highest weights
        remaining = n_samples - total_allocated
        sorted_indices = np.argsort(component_weights)[::-1]
        for i in range(remaining):
            samples_per_component[sorted_indices[i % len(sorted_indices)]] += 1
    elif total_allocated > n_samples:
        # Remove excess samples from components with lowest weights
        excess = total_allocated - n_samples
        sorted_indices = np.argsort(component_weights)
        for i in range(excess):
            idx = sorted_indices[i % len(sorted_indices)]
            if samples_per_component[idx] > 1:
                samples_per_component[idx] -= 1
    
    # Generate data from each component
    data = []
    
    for i in range(n_components):
        if samples_per_component[i] > 0:
            # Generate samples from this component with its specific std
            component_samples = np.random.normal(
                loc=means[i], 
                scale=stds[i], 
                size=(samples_per_component[i], 2)
            )
            data.append(component_samples)
    
    # Combine all samples
    data = np.vstack(data)
    
    # Shuffle the data
    np.random.shuffle(data)
    
    return data


class GMM2DDataset(Dataset):
    """
    PyTorch Dataset class for 2D GMM data.
    """
    
    def __init__(self, n_samples: int = 10000, n_components: int = 8, 
                 radius: float = 5.0, std: float = 0.2, 
                 cache_data: bool = True, complex_data: bool = False,
                 base_radius: float = 3.0, max_radius: float = 8.0,
                 min_std: float = 0.1, max_std: float = 0.4):
        """
        Initialize the GMM dataset.
        
        Args:
            n_samples: Number of samples in the dataset
            n_components: Number of Gaussian components
            radius: Radius of the circle where means are placed (for simple GMM)
            std: Standard deviation of each Gaussian component (for simple GMM)
            cache_data: Whether to cache the generated data
            complex_data: Whether to use complex GMM with varying parameters
            base_radius: Minimum radius for component placement (for complex GMM)
            max_radius: Maximum radius for component placement (for complex GMM)
            min_std: Minimum standard deviation (for complex GMM)
            max_std: Maximum standard deviation (for complex GMM)
        """
        self.n_samples = n_samples
        self.n_components = n_components
        self.radius = radius
        self.std = std
        self.cache_data = cache_data
        self.complex_data = complex_data
        self.base_radius = base_radius
        self.max_radius = max_radius
        self.min_std = min_std
        self.max_std = max_std
        
        if cache_data:
            if complex_data:
                self.data = generate_complex_gmm_data(
                    n_samples, n_components, base_radius, max_radius, min_std, max_std
                )
            else:
                self.data = generate_gmm_data(n_samples, n_components, radius, std)
            self.data = torch.FloatTensor(self.data)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.cache_data:
            return self.data[idx]
        else:
            # Generate data on the fly (not recommended for large datasets)
            if self.complex_data:
                sample = generate_complex_gmm_data(
                    1, self.n_components, self.base_radius, self.max_radius, 
                    self.min_std, self.max_std
                )
            else:
                sample = generate_gmm_data(1, self.n_components, self.radius, self.std)
            return torch.FloatTensor(sample[0])


def get_dataloader(batch_size: int = 128, train: bool = True, 
                   n_samples: int = 10000, n_components: int = 8,
                   radius: float = 5.0, std: float = 0.2,
                   shuffle: bool = True, num_workers: int = 0,
                   complex_data: bool = False, base_radius: float = 3.0,
                   max_radius: float = 8.0, min_std: float = 0.1,
                   max_std: float = 0.4) -> DataLoader:
    """
    Create a PyTorch DataLoader for the GMM dataset.
    
    Args:
        batch_size: Batch size for the DataLoader
        train: Whether this is for training (affects shuffle behavior)
        n_samples: Number of samples in the dataset
        n_components: Number of Gaussian components
        radius: Radius of the circle where means are placed (for simple GMM)
        std: Standard deviation of each Gaussian component (for simple GMM)
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        complex_data: Whether to use complex GMM with varying parameters
        base_radius: Minimum radius for component placement (for complex GMM)
        max_radius: Maximum radius for component placement (for complex GMM)
        min_std: Minimum standard deviation (for complex GMM)
        max_std: Maximum standard deviation (for complex GMM)
    
    Returns:
        PyTorch DataLoader
    """
    dataset = GMM2DDataset(
        n_samples=n_samples,
        n_components=n_components,
        radius=radius,
        std=std,
        cache_data=True,
        complex_data=complex_data,
        base_radius=base_radius,
        max_radius=max_radius,
        min_std=min_std,
        max_std=max_std
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if train else False,
        num_workers=num_workers,
        drop_last=train
    )


def plot_gmm_data(data: np.ndarray, title: str = "GMM Dataset", 
                  save_path: Optional[str] = None) -> None:
    """
    Plot the GMM data to visualize the distribution.
    
    Args:
        data: numpy array of shape (n_samples, 2)
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.6, s=1)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    # Test the GMM dataset generation and visualization
    
    print("Generating GMM dataset...")
    
    # Generate sample data
    n_samples = 5000
    n_components = 8
    radius = 5.0
    std = 0.2
    
    # Generate data
    data = generate_gmm_data(n_samples, n_components, radius, std)
    print(f"Generated {data.shape[0]} samples with shape {data.shape}")
    
    # Create dataset and dataloader
    dataset = GMM2DDataset(n_samples=n_samples, n_components=n_components, 
                           radius=radius, std=std)
    dataloader = get_dataloader(batch_size=128, train=True, n_samples=n_samples,
                               n_components=n_components, radius=radius, std=std)
    
    # Test dataloader
    print(f"Dataset length: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Get a sample batch
    sample_batch = next(iter(dataloader))
    print(f"Sample batch shape: {sample_batch.shape}")
    print(f"Sample batch dtype: {sample_batch.dtype}")
    
    # Plot the data
    print("Plotting GMM dataset...")
    plot_gmm_data(data, title=f"GMM Dataset ({n_components} components, radius={radius})")
    
    # Test complex GMM
    print("\nGenerating complex GMM dataset...")
    complex_data = generate_complex_gmm_data(n_samples, n_components=12)
    print(f"Generated complex GMM with {complex_data.shape[0]} samples")
    plot_gmm_data(complex_data, title="Complex GMM Dataset (varying parameters)")
    
    print("GMM dataset test completed successfully!") 