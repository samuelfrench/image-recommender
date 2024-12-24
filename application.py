from flask import Flask, jsonify, request, session
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
import uuid

# Load Image Embeddings
image_embeddings = np.load('image_embeddings.npy', allow_pickle=True).item()
image_names = list(image_embeddings.keys())
embeddings = np.array(list(image_embeddings.values()))

# Initialize Flask app
application = Flask(__name__)
application.secret_key = 'your_secret_key'  # Set a secure key for sessions
CORS(application)

# User-specific storage
user_sessions = {}

# Helper to initialize user session
def initialize_session():
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    user_sessions[session_id] = {
        'model': create_model(),
        'feedback_data': [],
        'viewed_images': set()
    }
    return session_id

# Create a new model for each user session
def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(embeddings.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Get user-specific storage
def get_user_data():
    session_id = session.get('session_id')
    if not session_id or session_id not in user_sessions:
        session_id = initialize_session()
    return user_sessions[session_id]

# Get random image
def get_random_image(viewed_images):
    remaining_images = list(set(image_names) - viewed_images)
    if not remaining_images:
        return None
    next_image = random.choice(remaining_images)
    viewed_images.add(next_image)
    return next_image

# Recommend next image
def recommend_next_image(model, viewed_images, feedback_data):
    rated_images = [image_names[idx] for idx, _ in enumerate(feedback_data)]
    candidate_images = [name for name in image_names if name not in rated_images and name not in viewed_images]

    if not candidate_images:
        return None

    candidate_embeddings = np.array([image_embeddings[name] for name in candidate_images])
    predictions = model.predict(candidate_embeddings)
    best_index = np.argmax(predictions)
    next_image = candidate_images[best_index]
    viewed_images.add(next_image)  # Mark as viewed
    return next_image

# Routes
@application.route('/get-image', methods=['GET'])
def get_image():
    user_data = get_user_data()
    current_image = get_random_image(user_data['viewed_images'])
    if not current_image:
        return jsonify({'message': 'No more images to display'}), 200
    return jsonify({'image': current_image})

@application.route('/send-feedback', methods=['POST'])
def send_feedback():
    user_data = get_user_data()
    model = user_data['model']
    feedback_data = user_data['feedback_data']

    data = request.json
    image_name = data['image']
    like = data['like']  # 1 for thumbs up, 0 for thumbs down

    embedding = image_embeddings[image_name]
    feedback_data.append((embedding, like))

    # Train the user-specific model
    if len(feedback_data) >= 5:  # Train only if enough data is collected
        X = np.array([item[0] for item in feedback_data])
        y = np.array([item[1] for item in feedback_data])
        model.fit(X, y, epochs=1, batch_size=4, verbose=1)

    return jsonify({'status': 'feedback received'})

@application.route('/recommend-next', methods=['GET'])
def recommend():
    user_data = get_user_data()
    next_image = recommend_next_image(user_data['model'], user_data['viewed_images'], user_data['feedback_data'])
    if not next_image:
        return jsonify({'message': 'No more images to recommend'}), 200
    return jsonify({'image': next_image})

if __name__ == '__main__':
    application.run(debug=True)
