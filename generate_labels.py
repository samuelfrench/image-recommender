import os
import json  # For saving the labels
from tqdm import tqdm
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Initialize MobileNetV2 model
mobilenet_model = MobileNetV2(weights='imagenet')

def generate_label_to_images(output_file='label_to_images.json'):
    """
    Process images and save label-to-image mappings to a JSON file.
    """
    print("Processing images to generate label-to-image mappings...")
    pics_dir = 'reddit-pics'
    if not os.path.exists(pics_dir):
        print(f"Error: {pics_dir} directory not found")
        return

    label_to_images = {}

    for image_name in tqdm(os.listdir(pics_dir)):
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        try:
            image_path = os.path.join(pics_dir, image_name)
            img = load_img(image_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array[np.newaxis, ...])

            predictions = mobilenet_model.predict(img_array)
            decoded_predictions = decode_predictions(predictions, top=5)[0]

            for (label, description, confidence) in decoded_predictions:
                if confidence > 0.1:  # Only store predictions with >10% confidence
                    if label not in label_to_images:
                        label_to_images[label] = []
                    label_to_images[label].append(image_name)
        except Exception as e:
            print(f"Error processing {image_name}: {e}")

    # Save the label-to-images mapping to a JSON file
    with open(output_file, 'w') as f:
        json.dump(label_to_images, f, indent=4)
    
    print(f"Saved label-to-image mappings to {output_file}")

if __name__ == '__main__':
    generate_label_to_images()