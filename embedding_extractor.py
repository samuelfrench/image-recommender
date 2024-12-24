import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

# Path to your folder of images
IMAGE_FOLDER = '/home/sam/g-images'

# Load the pre-trained ResNet model
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_embedding(image_path):
    """Extracts a feature embedding for a single image."""
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    embedding = model.predict(img_array)
    return embedding.flatten()

# Extract embeddings for all images
image_embeddings = {}
for image_file in os.listdir(IMAGE_FOLDER):
    if image_file.endswith(('jpg', 'jpeg', 'png', 'gif')):
        image_path = os.path.join(IMAGE_FOLDER, image_file)
        embedding = extract_embedding(image_path)
        image_embeddings[image_file] = embedding

# Save embeddings for later use (optional)
np.save('image_embeddings.npy', image_embeddings)

