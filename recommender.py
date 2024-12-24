import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load Image Embeddings
image_embeddings = np.load('image_embeddings.npy', allow_pickle=True).item()  # Assuming a dictionary {image_name: embedding}
image_names = list(image_embeddings.keys())
embeddings = np.array(list(image_embeddings.values()))

print(f"Loaded {len(image_names)} images.")

# Define a Simple Neural Network Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(embeddings.shape[1],)),  # Embedding size as input
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Probability output
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# In-Memory Storage for Feedback
feedback_data = []

def get_next_image():
    """Get a random image for the user to rate."""
    return random.choice(image_names)

def collect_feedback(image_name, like):
    """
    Collect user feedback.
    :param image_name: Name of the image
    :param like: 1 for thumbs up, 0 for thumbs down
    """
    embedding = image_embeddings[image_name]
    feedback_data.append((embedding, like))
    print(f"Collected feedback for {image_name}: {'Thumbs Up' if like else 'Thumbs Down'}")

def train_model():
    """Train the model on the collected feedback."""
    if len(feedback_data) < 10:  # Minimum feedback for training
        print("Not enough feedback to train the model.")
        return

    # Prepare training data
    X = np.array([item[0] for item in feedback_data])  # Embeddings
    y = np.array([item[1] for item in feedback_data])  # Labels

    # Train the model
    model.fit(X, y, epochs=5, batch_size=4, verbose=1)
    print("Model trained with user feedback.")

def recommend_next_image():
    """
    Recommend the next image based on the trained model.
    :return: Image name of the recommended image
    """
    # Get names of images already rated
    rated_images = [image_names[idx] for idx, _ in enumerate(feedback_data)]

    # Filter out rated images
    candidate_images = [name for name in image_names if name not in rated_images]
    if not candidate_images:
        print("No more images to recommend.")
        return None

    candidate_embeddings = np.array([image_embeddings[name] for name in candidate_images])

    # Predict scores for the remaining candidates
    predictions = model.predict(candidate_embeddings)
    best_index = np.argmax(predictions)
    return candidate_images[best_index]

def run_system():
    while True:
        current_image = get_next_image()
        print(f"Current image: {current_image}")

        # Simulate showing the image (replace with actual image display in a web app)
        user_input = input("Thumbs up (1) or thumbs down (0)? Enter 'q' to quit: ")
        if user_input.lower() == 'q':
            break

        like = int(user_input)
        collect_feedback(current_image, like)

        # Retrain the model
        train_model()

        # Recommend the next image
        next_image = recommend_next_image()
        print(f"Next recommended image: {next_image}")

if __name__ == "__main__":
    run_system()

