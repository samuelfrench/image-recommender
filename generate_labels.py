import os
import json
from tqdm import tqdm
import numpy as np
from tensorflow.keras.applications import EfficientNetV2L, ResNet152V2, MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficient_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobile_preprocess

# Load ImageNet class names from a JSON file
with open('imagenet_classes.json', 'r') as f:
    IMAGENET_CLASSES = json.load(f)

# Model configurations with correct input sizes
MODEL_CONFIGS = {
    'efficient': {
        'model': EfficientNetV2L(weights='imagenet'),
        'preprocess': efficient_preprocess,
        'size': (480, 480)
    },
    'resnet': {
        'model': ResNet152V2(weights='imagenet'),
        'preprocess': resnet_preprocess,
        'size': (224, 224)
    },
    'mobile': {
        'model': MobileNetV2(weights='imagenet'),
        'preprocess': mobile_preprocess,
        'size': (224, 224)
    }
}

def manual_decode_predictions(preds, top=5):
    """
    Decode predictions manually using ImageNet class indices.
    Handle missing indices gracefully.
    """
    try:
        top_indices = np.argsort(preds[0])[-top:][::-1]
        results = []
        for i in top_indices:
            str_i = str(i)
            if str_i in IMAGENET_CLASSES:
                results.append((i, IMAGENET_CLASSES[str_i], preds[0][i]))
        return results
    except Exception as e:
        print(f"Error decoding predictions: {e}")
        return []

def process_image(image_path, target_size):
    """Process image with correct target size"""
    try:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        return [
            img_array,  # Original
            np.fliplr(img_array),  # Horizontal flip
            img_array[:, ::-1],  # Vertical flip
        ]
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def get_predictions(image_array, model_name):
    """Get predictions from a single model with improved error handling"""
    try:
        config = MODEL_CONFIGS[model_name]
        processed = config['preprocess'](image_array)
        preds = config['model'].predict(processed[np.newaxis, ...], verbose=0)
        return manual_decode_predictions(preds, top=10)
    except Exception as e:
        print(f"Error getting predictions: {e}")
        return []

def generate_label_to_images(output_file='label_to_images_with_confidence.json'):
    print("Processing images with ensemble models...")
    pics_dir = 'reddit-pics'
    if not os.path.exists(pics_dir):
        print(f"Error: {pics_dir} directory not found")
        return

    label_to_images = {}
    processed_count = 0

    for image_name in tqdm(os.listdir(pics_dir)):
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        try:
            image_path = os.path.join(pics_dir, image_name)
            all_predictions = {}

            # Process with each model
            for model_name, config in MODEL_CONFIGS.items():
                augs = process_image(image_path, config['size'])
                if not augs:
                    continue

                for aug in augs:
                    predictions = get_predictions(aug, model_name)
                    for (_, label, conf) in predictions:
                        if label not in all_predictions:
                            all_predictions[label] = []
                        all_predictions[label].append(conf)

            # Aggregate predictions and store confidence levels
            for label, confidences in all_predictions.items():
                avg_conf = float(np.mean(confidences))
                if avg_conf > 0.000001:  # Confidence threshold
                    if label not in label_to_images:
                        label_to_images[label] = []
                    label_to_images[label].append({
                        "image": image_name,
                        "confidence": avg_conf
                    })

            processed_count += 1
            if processed_count % 10 == 0:  # Save progress frequently
                with open(output_file, 'w') as f:
                    json.dump(label_to_images, f, indent=4)

        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            continue

    # Final save
    with open(output_file, 'w') as f:
        json.dump(label_to_images, f, indent=4)
    
    print(f"Processed {processed_count} images")
    print(f"Found {len(label_to_images)} unique labels")

if __name__ == '__main__':
    generate_label_to_images()