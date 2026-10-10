import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler
from PIL import Image
import requests
from io import BytesIO
import time
import os

class PCACustom:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None
        self.explained_variance_ratio = None
        
    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvectors by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store explained variance
        self.explained_variance = eigenvalues
        self.explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        
        # Select top n_components
        if self.n_components is not None:
            self.components = eigenvectors[:, :self.n_components]
        else:
            self.components = eigenvectors
            
        return self
    
    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        return np.dot(X_transformed, self.components.T) + self.mean

# Test PCA with sample data
print("="*50)
print("PCA TESTING")
print("="*50)

# Generate test data
np.random.seed(42)
X_test = np.random.randn(100, 3) * 10
X_test[:, 1] = X_test[:, 0] * 2 + np.random.randn(100) * 2
X_test[:, 2] = X_test[:, 0] * 0.5 + np.random.randn(100) * 5

# Custom PCA
pca_custom = PCACustom(n_components=2)
X_custom = pca_custom.fit_transform(X_test)

# Sklearn PCA
pca_sklearn = SklearnPCA(n_components=2)
X_sklearn = pca_sklearn.fit_transform(X_test)

print("Custom PCA explained variance ratio:", pca_custom.explained_variance_ratio)
print("Sklearn PCA explained variance ratio:", pca_sklearn.explained_variance_ratio_)
print("Difference in components:", np.max(np.abs(np.abs(pca_custom.components) - np.abs(pca_sklearn.components_.T))))

def load_image():
    """Load image from local file or download from web"""
    # First try to load local image
    local_paths = [
        "HW4/Toyota-Chaser-JZX100-10.jpg",
    ]
    
    for path in local_paths:
        try:
            if os.path.exists(path):
                img = Image.open(path)
                print(f"Loaded image from: {path}")
                break
        except:
            continue
    else:
        try:
            url = "https://nextlevelautomotive.eu/wp-content/uploads/2020/09/Toyota-Chaser-JZX100-1.jpg"
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content))
            print("Successfully downloaded image")
        except Exception as e:
            print(f"Could not load image: {e}")
            print("Using synthetic image instead...")
            img_array = np.zeros((256, 256, 3), dtype=np.uint8)
            # Create a gradient pattern
            for i in range(256):
                for j in range(256):
                    img_array[i, j, 0] = i
                    img_array[i, j, 1] = j
                    img_array[i, j, 2] = (i + j) // 2
            # Add some pattern
            for i in range(0, 256, 20):
                img_array[i:i+5, :, :] = 255
                img_array[:, i:i+5, :] = 255
            return img_array
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Ensure RGB
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array]*3, axis=2)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    
    # Resize if needed (minimum 256x256)
    if img_array.shape[0] < 256 or img_array.shape[1] < 256:
        img_array = np.array(Image.fromarray(img_array).resize((256, 256)))
        print(f"Resized image to: {img_array.shape}")
    
    return img_array

def compress_image_with_pca(image, k_values):
    """Compress image using PCA for each channel"""
    height, width, channels = image.shape
    compressed_images = {}
    
    for k in k_values:
        print(f"Compressing with K={k}...")
        reconstructed_channels = []
        compression_ratios = []
        
        for channel in range(channels):
            # Get channel data
            channel_data = image[:, :, channel]
            
            # Reshape to 2D: (height, width) -> (height, width)
            X = channel_data.astype(np.float64)
            
            # Apply PCA
            pca = PCACustom(n_components=k)
            X_transformed = pca.fit_transform(X)
            X_reconstructed = pca.inverse_transform(X_transformed)
            
            # Clip values to valid range
            X_reconstructed = np.clip(X_reconstructed, 0, 255).astype(np.uint8)
            reconstructed_channels.append(X_reconstructed)
            
            # Calculate compression ratio
            original_size = X.shape[0] * X.shape[1]
            compressed_size = X_transformed.shape[0] * X_transformed.shape[1] + pca.components.shape[0] * pca.components.shape[1]
            compression_ratios.append(original_size / compressed_size)
        
        # Combine channels
        reconstructed_image = np.stack(reconstructed_channels, axis=2)
        compressed_images[k] = reconstructed_image
        
        print(f"  Average compression ratio: {np.mean(compression_ratios):.2f}")
        
    return compressed_images

# Load and compress image
print("\n" + "="*50)
print("IMAGE COMPRESSION WITH PCA")
print("="*50)

image = load_image()
print(f"Image shape: {image.shape}")

# Test different K values
k_values = [10, 30, 50]
compressed_images = compress_image_with_pca(image, k_values)

# Display results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(image)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis('off')

for idx, k in enumerate(k_values):
    row = 0 if idx < 2 else 1
    col = (idx + 1) if idx < 2 else (idx - 1)
    axes[row, col].imshow(compressed_images[k])
    axes[row, col].set_title(f"K={k}")
    axes[row, col].axis('off')

# Show difference images
for idx, k in enumerate(k_values):
    row = 1
    col = idx
    diff = np.abs(image.astype(np.float32) - compressed_images[k].astype(np.float32))
    axes[row, col].imshow(diff.astype(np.uint8))
    axes[row, col].set_title(f"Diff K={k}")
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('pca_compression_results.png', dpi=150)
plt.show()

class KMeansCustom:
    def __init__(self, n_clusters=3, max_iter=300, random_state=None, verbose=True):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        self.verbose = verbose
        
    def initialize_centroids(self, X):
        """Initialize centroids using random selection"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        # Randomly select k samples as initial centroids
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        return X[indices].copy()
    
    def assign_clusters(self, X):
        """Assign each sample to the nearest centroid"""
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.linalg.norm(X - self.centroids[k], axis=1)
        return np.argmin(distances, axis=1)
    
    def update_centroids(self, X, labels):
        """Update centroids based on mean of assigned points"""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            if np.sum(labels == k) > 0:
                new_centroids[k] = np.mean(X[labels == k], axis=0)
            else:
                # If cluster is empty, reinitialize randomly
                new_centroids[k] = X[np.random.choice(X.shape[0])]
        return new_centroids
    
    def fit(self, X):
        """Main K-Means algorithm"""
        # Initialize centroids
        self.centroids = self.initialize_centroids(X)
        
        for iteration in range(self.max_iter):
            # Assign clusters
            old_labels = self.labels
            self.labels = self.assign_clusters(X)
            
            # Update centroids
            new_centroids = self.update_centroids(X, self.labels)
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids, rtol=1e-4):
                if self.verbose:
                    print(f"Converged at iteration {iteration+1}")
                break
            
            self.centroids = new_centroids
        
        # Calculate inertia
        self.inertia_ = 0
        for k in range(self.n_clusters):
            if np.sum(self.labels == k) > 0:
                self.inertia_ += np.sum((X[self.labels == k] - self.centroids[k]) ** 2)
        
        return self
    
    def predict(self, X):
        if self.centroids is None:
            raise ValueError("Model not fitted yet!")
        return self.assign_clusters(X)
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels

print("\n" + "="*50)
print("K-MEANS ON IRIS DATASET")
print("="*50)

# Load Iris dataset
iris = load_iris()
X_iris = iris.data[:, [2, 3]]  # Petal length and petal width
y_true = iris.target

# Test different K values
k_values_to_test = [2, 3, 4, 5]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, k in enumerate(k_values_to_test):
    print(f"\nTesting K={k}")
    
    # Custom K-Means
    kmeans_custom = KMeansCustom(n_clusters=k, max_iter=300, random_state=42, verbose=True)
    labels_custom = kmeans_custom.fit_predict(X_iris)
    
    # Plot results
    ax = axes[idx]
    scatter = ax.scatter(X_iris[:, 0], X_iris[:, 1], 
                        c=labels_custom, cmap='viridis', 
                        alpha=0.7, s=50)
    ax.scatter(kmeans_custom.centroids[:, 0], kmeans_custom.centroids[:, 1], 
              c='red', marker='X', s=200, linewidths=3, edgecolors='black')
    ax.set_xlabel('Petal Length')
    ax.set_ylabel('Petal Width')
    ax.set_title(f'K={k}, Inertia={kmeans_custom.inertia_:.2f}')
    ax.grid(True, alpha=0.3)
    
    # Print cluster information
    for cluster in range(k):
        cluster_size = np.sum(labels_custom == cluster)
        print(f"  Cluster {cluster}: {cluster_size} samples")

# Also show comparison with true labels for K=3
kmeans_3 = KMeansCustom(n_clusters=3, max_iter=300, random_state=42, verbose=True)
labels_3 = kmeans_3.fit_predict(X_iris)

ax = axes[4]
scatter = ax.scatter(X_iris[:, 0], X_iris[:, 1], 
                    c=labels_3, cmap='viridis', 
                    alpha=0.7, s=50)
ax.scatter(kmeans_3.centroids[:, 0], kmeans_3.centroids[:, 1], 
          c='red', marker='X', s=200, linewidths=3, edgecolors='black')
ax.set_xlabel('Petal Length')
ax.set_ylabel('Petal Width')
ax.set_title(f'K=3 (Custom)')
ax.grid(True, alpha=0.3)

# Show true labels for comparison
ax = axes[5]
scatter = ax.scatter(X_iris[:, 0], X_iris[:, 1], 
                    c=y_true, cmap='viridis', 
                    alpha=0.7, s=50)
ax.set_xlabel('Petal Length')
ax.set_ylabel('Petal Width')
ax.set_title('True Labels')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kmeans_iris_results.png', dpi=150)
plt.show()

# Elbow Method for optimal K
print("\n" + "="*50)
print("ELBOW METHOD FOR OPTIMAL K")
print("="*50)

inertias = []
k_range = range(1, 11)
for k in k_range:
    print(f"Testing K={k}...")
    kmeans = KMeansCustom(n_clusters=k, max_iter=300, random_state=42, verbose=False)
    kmeans.fit(X_iris)
    inertias.append(kmeans.inertia_)
    print(f"  Inertia: {kmeans.inertia_:.2f}")

plt.figure(figsize=(8, 6))
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.savefig('elbow_method.png', dpi=150)
plt.show()