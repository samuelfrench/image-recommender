# Image Recommendation System

This project uses **TensorFlow** to provide personalized image recommendations based on your previous selections of thumbs up and thumbs down.

## Features
- Dynamic image recommendations.
- Tracks user feedback to improve suggestions.
- Avoids showing the same image twice.

## How to Run

### 1. Uncompress Images
Download (https://drive.google.com/file/d/1KX559Ucd59_zLTQcUMJO20q6-dCW427o/view?usp=sharing) and uncompress reddit-pics.tar.xz to a folder named: s3/reddit-pics

### 2. Start the Backend
Install application/requirements.txt file

Generate a self-signed SSL certificate:
```bash
cd application
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
```

Run the following command to start the Flask backend:
```bash
python application/application.py
```

### 3. Start the Frontend
Serve the frontend files using Python's HTTP server:
```bash
cd s3
python -m http.server 8080
```

### 3. Access the Application
Open your browser and navigate to:
```
http://127.0.0.1:8080
```

Enjoy personalized image recommendations!
