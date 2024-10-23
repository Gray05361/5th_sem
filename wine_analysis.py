import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import random

def generate_random_mac():
    return ':'.join(['{:02x}'.format(random.randint(0, 255)) for _ in range(6)])

wine_data = load_wine()
data_features = wine_data.data
data_labels = wine_data.target

wine_dataframe = pd.DataFrame(data_features, columns=wine_data.feature_names)
wine_dataframe['class'] = data_labels

print("First 5 rows of the dataset:")
print(wine_dataframe.head())
print("\nDataset Description:")
print(wine_dataframe.describe())

random_mac = generate_random_mac()
print(f"\nGenerated MAC Address: {random_mac}")

scaler_instance = StandardScaler()
scaled_data = scaler_instance.fit_transform(data_features)

pca_transformer = PCA(n_components=2)
pca_result = pca_transformer.fit_transform(scaled_data)

plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=data_labels, cmap='viridis', edgecolor='k', s=50)
plt.title('PCA Visualization of Wine Dataset')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Class')
plt.grid()
plt.show()

tsne_transformer = TSNE(n_components=2, random_state=42)
tsne_result = tsne_transformer.fit_transform(scaled_data)

plt.figure(figsize=(10, 6))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=data_labels, cmap='viridis', edgecolor='k', s=50)
plt.title('t-SNE Visualization of Wine Dataset')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label='Class')
plt.grid()
plt.show()

kmeans_pca = KMeans(n_clusters=3, random_state=42)
pca_clusters = kmeans_pca.fit_predict(pca_result)

pca_silhouette = silhouette_score(pca_result, pca_clusters)
print(f'Silhouette Score (PCA Reduced Data): {pca_silhouette:.2f}')

kmeans_tsne = KMeans(n_clusters=3, random_state=42)
tsne_clusters = kmeans_tsne.fit_predict(tsne_result)

tsne_silhouette = silhouette_score(tsne_result, tsne_clusters)
print(f'Silhouette Score (t-SNE Reduced Data): {tsne_silhouette:.2f}')

kmeans_original = KMeans(n_clusters=3, random_state=42)
original_clusters = kmeans_original.fit_predict(scaled_data)

original_silhouette = silhouette_score(scaled_data, original_clusters)
print(f'Silhouette Score (Original Data): {original_silhouette:.2f}')

