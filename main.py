import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm


# Function to implement k-means++ initialization
def kmeans_plus_plus_init(data, k):
    # Choose first centroid randomly
    centroids = [data[np.random.randint(len(data))]]

    # Choose remaining centroids
    for _ in range(1, k):
        # Calculate distances from points to nearest existing centroid
        distances = np.array([min([np.linalg.norm(point - cent) for cent in centroids]) for point in data])

        # Choose next centroid with probability proportional to distance squared
        probabilities = distances ** 2 / np.sum(distances ** 2)
        cumulative_probs = np.cumsum(probabilities)
        r = np.random.random()

        for j, p in enumerate(cumulative_probs):
            if r < p:
                centroids.append(data[j])
                break

    return np.array(centroids)


# Function to implement k-means algorithm
def kmeans(data, k, max_iterations=100):
    # Initialize centroids using k-means++ method
    centroids = kmeans_plus_plus_init(data, k)

    for _ in range(max_iterations):
        # Assign each point to nearest centroid
        distances = np.array([[np.linalg.norm(point - centroid) for centroid in centroids] for point in data])
        clusters = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.array([data[clusters == i].mean(axis=0) if np.sum(clusters == i) > 0
                                  else centroids[i] for i in range(k)])

        # Check for convergence
        if np.allclose(centroids, new_centroids, rtol=1e-5):
            break

        centroids = new_centroids

    return clusters, centroids


# Function to plot clustering results
def plot_clusters(data, clusters, centroids, k, dark_mode=False):
    plt.figure(figsize=(10, 8))

    # Set up dark or light mode
    if dark_mode:
        plt.style.use('dark_background')
        edge_color = 'gray'
        alpha = 0.8
    else:
        plt.style.use('default')
        edge_color = 'w'
        alpha = 0.7

    # Create color map with vibrant distinct colors
    colors = cm.viridis(np.linspace(0, 1, k))

    # Plot data points
    for i in range(k):
        cluster_points = data[clusters == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i],
                    s=70, alpha=alpha, edgecolor=edge_color, linewidth=0.5, label=f'Cluster {i + 1}')

    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], s=150, marker='*',
                color='white' if dark_mode else 'black', edgecolor='red', linewidth=1.5, label='Centroids')

    # Add titles and labels
    plt.title(f'K-Means Clustering (k={k})', fontsize=16, fontweight='bold')
    plt.xlabel('X1', fontsize=12)
    plt.ylabel('X2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()

    return plt


# Function to find best clustering from multiple runs
def find_best_clustering(data, k, runs=10):
    best_inertia = float('inf')
    best_clusters = None
    best_centroids = None

    for _ in range(runs):
        clusters, centroids = kmeans(data, k)

        # Calculate inertia (sum of squared distances to centroids)
        inertia = 0
        for i in range(k):
            cluster_points = data[clusters == i]
            if len(cluster_points) > 0:
                inertia += np.sum([np.linalg.norm(point - centroids[i]) ** 2 for point in cluster_points])

        if inertia < best_inertia:
            best_inertia = inertia
            best_clusters = clusters
            best_centroids = centroids

    return best_clusters, best_centroids


# Main function
def main():
    # Load dataset
    df = pd.read_csv('kmeans_blobs.csv')  # Adjust filename if needed
    data = df[['x1', 'x2']].values

    # Normalize data
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    data_normalized = (data - data_mean) / data_std

    # Run k-means for k=2 (multiple times and select best)
    np.random.seed(42)
    clusters_k2, centroids_k2 = find_best_clustering(data_normalized, k=2, runs=10)

    # Run k-means for k=3 (multiple times and select best)
    np.random.seed(42)
    clusters_k3, centroids_k3 = find_best_clustering(data_normalized, k=3, runs=10)

    # Convert centroids back to original scale for plotting
    centroids_k2_original = centroids_k2 * data_std + data_mean
    centroids_k3_original = centroids_k3 * data_std + data_mean

    # Create plots
    plot_k2 = plot_clusters(data, clusters_k2, centroids_k2_original, k=2)
    plot_k2.savefig('kmeans_k2.png', dpi=300, bbox_inches='tight')

    plot_k3 = plot_clusters(data, clusters_k3, centroids_k3_original, k=3, dark_mode=False)
    plot_k3.savefig('kmeans_k3.png', dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    main()
