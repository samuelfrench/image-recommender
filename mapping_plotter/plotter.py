import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from umap import UMAP

# Load embeddings
embeddings = np.load('../high_fidelity_image_embeddings.npy', allow_pickle=True).item()
image_names = list(embeddings.keys())
embedding_vectors = np.array(list(embeddings.values()))

# Reduce dimensions to 2D using UMAP
reducer = UMAP(n_neighbors=30, n_components=2, random_state=42)  # Larger n_neighbors
reduced_embeddings = reducer.fit_transform(embedding_vectors)

# Path to your image directory
IMAGE_FOLDER = '../reddit-pics'  # Use resized images

# Helper function to load and resize an image for thumbnails
def load_thumbnail(image_path, zoom=0.05):  # Increase thumbnail size
    img = plt.imread(image_path)
    return OffsetImage(img, zoom=zoom)

print("Plot embeddings")
# Plot embeddings
fig, ax = plt.subplots(figsize=(30, 30), dpi=150)  # Larger canvas
ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.1, s=5, c='blue')  # Faint scatter points

print("Adding image thumbnails")
# Add image thumbnails
for i, image_name in enumerate(image_names[:100]):  # Limit to 100 images
    image_path = os.path.join(IMAGE_FOLDER, image_name)
    if os.path.exists(image_path):
        thumbnail = load_thumbnail(image_path, zoom=0.05)  # Increase zoom for visibility
        ab = AnnotationBbox(thumbnail, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                            frameon=False)
        ax.add_artist(ab)

# Customize plot appearance
ax.set_title("High Fidelity Image Embeddings with Thumbnails (Optimized)", fontsize=20)
ax.set_xlabel("Dimension 1", fontsize=14)
ax.set_ylabel("Dimension 2", fontsize=14)
ax.grid(False)  # Turn off grid for clarity

# Save plot as high-resolution PNG
output_file = "high_fidelity_image_embeddings.png"
plt.savefig(output_file, format='png', bbox_inches='tight')
plt.close()

print(f"Visualization saved to {output_file}")
