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
import random
import uuid
import os
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
image_embeddings = np.load('image_embeddings.npy', allow_pickle=True).item()
image_names = list(image_embeddings.keys())
embeddings = np.array(list(image_embeddings.values()))

# Store indexed results for each label
label_to_images = defaultdict(list)

# Initialize Flask app
application = Flask(__name__)
CORS(application, 
     supports_credentials=True,
     resources={r"/*": {
         "origins": ["http://localhost:8080"],
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

@application.route('/analyze-image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    image_file = request.files['image']
    image_path = os.path.join('/tmp', image_file.filename)
    image_file.save(image_path)

    try:
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = mobilenet_model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=5)[0]

        image_name = image_file.filename
        results = [
            {"label": label, "description": description, "confidence": float(confidence)}
            for (label, description, confidence) in decoded_predictions
        ]
        for prediction in results:
            label_to_images[prediction["label"]].append(image_name)

        return jsonify({'predictions': results})
    except Exception as e:
        return jsonify({'error': f"Failed to analyze image: {str(e)}"}), 500
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

@application.route('/get-similar-image/<label>', methods=['GET'])
def get_similar_image(label):
    session = get_or_create_session(request.cookies.get('session_id'))
    
    similar_images = label_to_images.get(label, [])
    if not similar_images:
        return jsonify({'message': 'No similar images found'}), 200

    unviewed_images = list(set(similar_images) - session.viewed_images)
    if not unviewed_images:
        return jsonify({'message': 'No unviewed similar images found'}), 200

    random_image = random.choice(unviewed_images)
    session.viewed_images.add(random_image)
    
    response = jsonify({'image': random_image})
    response.set_cookie('session_id', session.id)
    return response

if __name__ == '__main__':
    application.run(debug=True)