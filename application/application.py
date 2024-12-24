from collections import defaultdict
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm
import random
import uuid
import os
import json
from datetime import datetime, timedelta
from threading import Lock

class UserSession:
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.viewed_images = set()
        self.feedback_data = []
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.model = None

sessions = {}
session_lock = Lock()
SESSION_TIMEOUT = timedelta(hours=24)

def get_or_create_session(session_id=None):
    with session_lock:
        now = datetime.now()
        print(f"Incoming session_id: {session_id}")
        print(f"Current sessions: {list(sessions.keys())}")
        
        # Cleanup expired sessions
        expired = [sid for sid, session in sessions.items() 
                  if now - session.last_accessed > SESSION_TIMEOUT]
        for sid in expired:
            del sessions[sid]
            
        if session_id and session_id in sessions:
            session = sessions[session_id]
            session.last_accessed = now
        else:
            print("Creating new session")
            session = UserSession()
            session.model = create_model()
            sessions[session.id] = session
            
        return session

# Load Image Embeddings
image_embeddings = np.load('high_fidelity_image_embeddings.npy', allow_pickle=True).item()
image_names = list(image_embeddings.keys())
embeddings = np.array(list(image_embeddings.values()))

# Store indexed results for each label
label_to_images_with_confidence = {}

def initialize_label_to_images(file_path='label_to_images_with_confidence.json'):
    """
    Load the label-to-images dictionary from a JSON file.
    """
    global label_to_images_with_confidence

    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Please run the label generation script first.")
        return

    with open(file_path, 'r') as f:
        label_to_images_with_confidence = json.load(f)
    
    print(f"Loaded label-to-image mappings from {file_path}")
    print(f"Found {len(label_to_images_with_confidence)} unique labels")

# Initialize Flask app
application = Flask(__name__)
CORS(application, 
     supports_credentials=True,
     resources={r"/*": {
         "origins": ["*"],
         "methods": ["GET", "POST"],
         "allow_credentials": True
     }})

# Load pre-trained MobileNetV2 model
mobilenet_model = MobileNetV2(weights='imagenet')

def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(embeddings.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                 loss='binary_crossentropy', 
                 metrics=['accuracy'])
    return model

@application.route('/get-image', methods=['GET'])
def get_image():
    session = get_or_create_session(request.cookies.get('session_id'))
    
    remaining_images = list(set(image_names) - session.viewed_images)
    if not remaining_images:
        return jsonify({'message': 'No more images to display'}), 200
        
    current_image = random.choice(remaining_images)
    session.viewed_images.add(current_image)
    
    response = jsonify({'image': current_image})
    response.set_cookie('session_id', session.id)
    return response

@application.route('/send-feedback', methods=['POST'])
def send_feedback():
    session = get_or_create_session(request.cookies.get('session_id'))
    
    data = request.json
    image_name = data['image']
    like = data['like']
    
    embedding = image_embeddings[image_name]
    session.feedback_data.append((embedding, like))
    
    if len(session.feedback_data) >= 5:
        X = np.array([item[0] for item in session.feedback_data])
        y = np.array([item[1] for item in session.feedback_data])
        session.model.fit(X, y, epochs=1, batch_size=4, verbose=1)
    
    response = jsonify({'status': 'feedback received'})
    response.set_cookie('session_id', session.id)
    return response

@application.route('/recommend-next', methods=['GET'])
def recommend():
    session = get_or_create_session(request.cookies.get('session_id'))
    
    if not session.feedback_data:
        return get_image()
        
    candidate_images = [name for name in image_names 
                       if name not in session.viewed_images]
    
    if not candidate_images:
        return jsonify({'message': 'No more images to recommend'}), 200
        
    candidate_embeddings = np.array([image_embeddings[name] 
                                   for name in candidate_images])
    predictions = session.model.predict(candidate_embeddings)
    best_index = np.argmax(predictions)
    next_image = candidate_images[best_index]
    session.viewed_images.add(next_image)
    
    response = jsonify({'image': next_image})
    response.set_cookie('session_id', session.id)
    return response

def format_label(label):
    """Convert label ID to human-readable text"""
    return label.title()

@application.route('/analyze-image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    image_file = request.files['image']
    image_name = image_file.filename

    try:
        # Get predictions from pre-computed labels
        predictions = []
        for label, images in label_to_images_with_confidence.items():
            for entry in images:
                if entry["image"] == image_name:
                    predictions.append({
                        "label": label,
                        "description": format_label(label),
                        "confidence": entry["confidence"]
                    })
        
        if not predictions:
            return jsonify({'error': 'No predictions found for image'}), 404
            
        # Return the top 5 predictions sorted by confidence
        predictions = sorted(predictions, key=lambda x: x["confidence"], reverse=True)[:5]
        return jsonify({'predictions': predictions})
        
    except Exception as e:
        return jsonify({'error': f"Failed to analyze image: {str(e)}"}), 500

@application.route('/get-similar-image/<label>', methods=['GET'])
def get_similar_image(label):
    session = get_or_create_session(request.cookies.get('session_id'))
    
    similar_images = label_to_images_with_confidence.get(label, [])
    if not similar_images:
        return jsonify({'message': 'No similar images found'}), 200

    # Sort similar images by confidence, descending
    similar_images = sorted(similar_images, key=lambda x: x["confidence"], reverse=True)
    unviewed_images = [entry["image"] for entry in similar_images if entry["image"] not in session.viewed_images]
    
    if not unviewed_images:
        return jsonify({'message': 'No unviewed similar images found'}), 200

    random_image = unviewed_images[0]  # Get the highest-confidence unviewed image
    session.viewed_images.add(random_image)
    
    response = jsonify({'image': random_image})
    response.set_cookie('session_id', session.id)
    return response

if __name__ == '__main__':
    initialize_label_to_images()
    application.run(host='0.0.0.0', port=80, debug=True)