####################imports####################
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import json
import os
from datetime import datetime
####################imports####################

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Model and configurations
model = tf.keras.models.load_model(
    'saved_model.h5',
    custom_objects=None,
    compile=True,
    options=None
)

class_list = ['Alfred Enoch', 'Harry Potter', 'Hermione', 'Menna', 'Ron Weasley', 'Sama']
text_color = (206, 235, 135)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
thickness = 3

# Database simulation (in production, use a real database)
missing_persons_db = []
found_matches = []

def preprocess_image(image):
    """Preprocess image for model prediction"""
    image_resized = cv2.resize(image, (224, 224))
    img_array = tf.expand_dims(image_resized, 0)
    return img_array

def predict_face(image):
    """Make prediction on face image"""
    img_array = preprocess_image(image)
    predict = model.predict(img_array)
    predict_index = np.argmax(predict[0], axis=0)
    confidence = float(predict[0][predict_index])
    predicted_class = class_list[predict_index]
    return predicted_class, confidence

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/report_missing', methods=['POST'])
def report_missing():
    """Handle missing person report submission"""
    try:
        data = request.json
        
        # Extract guardian information
        guardian_info = {
            'name': data.get('guardian_name'),
            'age': data.get('guardian_age'),
            'national_id': data.get('guardian_national_id'),
            'phone': data.get('guardian_phone'),
            'relationship': data.get('relationship')
        }
        
        # Extract missing person information
        missing_info = {
            'name': data.get('missing_name'),
            'age': data.get('missing_age'),
            'national_id': data.get('missing_national_id'),
            'last_seen_location': data.get('last_seen_location'),
            'description': data.get('description'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Handle photo data
        photos = data.get('photos', [])
        processed_photos = []
        
        for photo_base64 in photos:
            # Decode base64 image
            img_data = base64.b64decode(photo_base64.split(',')[1])
            img = Image.open(io.BytesIO(img_data))
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Make prediction
            predicted_class, confidence = predict_face(img_cv)
            processed_photos.append({
                'predicted_identity': predicted_class,
                'confidence': confidence
            })
        
        # Create report entry
        report = {
            'id': len(missing_persons_db) + 1,
            'guardian': guardian_info,
            'missing_person': missing_info,
            'photos': processed_photos,
            'status': 'active'
        }
        
        missing_persons_db.append(report)
        
        return jsonify({
            'success': True,
            'message': 'Report submitted successfully',
            'report_id': report['id'],
            'predictions': processed_photos
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/search_by_photo', methods=['POST'])
def search_by_photo():
    """Search database by uploaded photo"""
    try:
        data = request.json
        photo_base64 = data.get('photo')
        
        # Decode and process image
        img_data = base64.b64decode(photo_base64.split(',')[1])
        img = Image.open(io.BytesIO(img_data))
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Make prediction
        predicted_class, confidence = predict_face(img_cv)
        
        # Search in database
        matches = []
        for report in missing_persons_db:
            for photo in report['photos']:
                if photo['predicted_identity'] == predicted_class and confidence > 0.7:
                    matches.append({
                        'report_id': report['id'],
                        'missing_person': report['missing_person'],
                        'guardian': report['guardian'],
                        'confidence': confidence,
                        'predicted_identity': predicted_class
                    })
        
        return jsonify({
            'success': True,
            'predicted_identity': predicted_class,
            'confidence': confidence,
            'matches': matches
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/live_recognition')
def live_recognition():
    """Stream live face recognition"""
    def generate_frames():
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            yield b'--frame\r\nContent-Type: text/plain\r\n\r\nCamera not available\r\n'
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Make prediction
            predicted_class, confidence = predict_face(frame)
            
            # Draw text on frame
            text = f"Detected: {predicted_class} ({confidence:.2%})"
            cv2.putText(
                frame,
                text,
                (50, 50),
                font,
                fontScale,
                text_color,
                thickness,
                cv2.LINE_AA
            )
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        cap.release()
    
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/get_missing_reports', methods=['GET'])
def get_missing_reports():
    """Get all missing person reports"""
    return jsonify({
        'success': True,
        'reports': missing_persons_db
    })

@app.route('/api/search_by_name', methods=['GET'])
def search_by_name():
    """Search missing persons by name"""
    name_query = request.args.get('name', '').lower()
    
    results = [
        report for report in missing_persons_db
        if name_query in report['missing_person']['name'].lower()
    ]
    
    return jsonify({
        'success': True,
        'results': results
    })

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    
    print("Starting Returno Face Recognition Server...")
    print("Server running on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
