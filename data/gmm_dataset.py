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


class GMM2DDataset(Dataset):
    """
    PyTorch Dataset class for 2D GMM data.
    """
    
    def __init__(self, n_samples: int = 10000, n_components: int = 8, 
                 radius: float = 5.0, std: float = 0.2, 
                 cache_data: bool = True):
        """
        Initialize the GMM dataset.
        
        Args:
            n_samples: Number of samples in the dataset
            n_components: Number of Gaussian components
            radius: Radius of the circle where means are placed
            std: Standard deviation of each Gaussian component
            cache_data: Whether to cache the generated data
        """
        self.n_samples = n_samples
        self.n_components = n_components
        self.radius = radius
        self.std = std
        self.cache_data = cache_data
        
        if cache_data:
            self.data = generate_gmm_data(n_samples, n_components, radius, std)
            self.data = torch.FloatTensor(self.data)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.cache_data:
            return self.data[idx]
        else:
            # Generate data on the fly (not recommended for large datasets)
            sample = generate_gmm_data(1, self.n_components, self.radius, self.std)
            return torch.FloatTensor(sample[0])


def get_dataloader(batch_size: int = 128, train: bool = True, 
                   n_samples: int = 10000, n_components: int = 8,
                   radius: float = 5.0, std: float = 0.2,
                   shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    """
    Create a PyTorch DataLoader for the GMM dataset.
    
    Args:
        batch_size: Batch size for the DataLoader
        train: Whether this is for training (affects shuffle behavior)
        n_samples: Number of samples in the dataset
        n_components: Number of Gaussian components
        radius: Radius of the circle where means are placed
        std: Standard deviation of each Gaussian component
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
    
    Returns:
        PyTorch DataLoader
    """
    dataset = GMM2DDataset(
        n_samples=n_samples,
        n_components=n_components,
        radius=radius,
        std=std,
        cache_data=True
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
    
    print("GMM dataset test completed successfully!") 