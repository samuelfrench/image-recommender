import os
import numpy as np
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

# Path to your folder of images
IMAGE_FOLDER = '/home/sam/image_recommender/reddit-pics'

# Load the pre-trained EfficientNetV2L model
base_model = EfficientNetV2L(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_embedding(image_path):
    """Extracts a high-fidelity feature embedding for a single image."""
    img = load_img(image_path, target_size=(512, 512))  # Higher resolution
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    embedding = model.predict(img_array, verbose=0)  # Suppress output for batch prediction
    return embedding.flatten()

# Extract embeddings for all images
image_embeddings = {}
for image_file in os.listdir(IMAGE_FOLDER):
    if image_file.lower().endswith(('jpg', 'jpeg', 'png', 'gif')):
        image_path = os.path.join(IMAGE_FOLDER, image_file)
        try:
            embedding = extract_embedding(image_path)
            image_embeddings[image_file] = embedding
            print(f"Processed {image_file}")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

# Save embeddings for later use
output_file = 'high_fidelity_image_embeddings.npy'
np.save(output_file, image_embeddings)

print(f"High-fidelity image embeddings saved to {output_file}")
