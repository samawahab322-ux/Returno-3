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
from flask_sqlalchemy import SQLAlchemy
####################imports####################

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

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

# Database Models
class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    guardian_id = db.Column(db.Integer, db.ForeignKey('guardian.id'), nullable=False)
    missing_person_id = db.Column(db.Integer, db.ForeignKey('missing_person.id'), nullable=False)
    status = db.Column(db.String(50), nullable=False, default='active')
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    guardian = db.relationship('Guardian', backref=db.backref('reports', lazy=True))
    missing_person = db.relationship('MissingPerson', backref=db.backref('reports', lazy=True))

class Guardian(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    national_id = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    relationship = db.Column(db.String(50), nullable=False)

class MissingPerson(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    national_id = db.Column(db.String(100), nullable=False)
    last_seen_location = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    predicted_identity = db.Column(db.String(100))
    confidence = db.Column(db.Float)

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

@app.route('/dashboard')
def dashboard():
    """Serve the dashboard page"""
    reports = Report.query.all()
    return render_template('dashboard.html', reports=reports)

@app.route('/api/report_missing', methods=['POST'])
def report_missing():
    """Handle missing person report submission"""
    try:
        data = request.json
        
        guardian = Guardian(
            name=data.get('guardian_name'),
            age=data.get('guardian_age'),
            national_id=data.get('guardian_national_id'),
            phone=data.get('guardian_phone'),
            relationship=data.get('relationship')
        )
        db.session.add(guardian)
        db.session.commit()

        missing_person = MissingPerson(
            name=data.get('missing_name'),
            age=data.get('missing_age'),
            national_id=data.get('missing_national_id'),
            last_seen_location=data.get('last_seen_location'),
            description=data.get('description')
        )
        
        photos = data.get('photos', [])
        predicted_identity = None
        confidence = 0.0

        if photos:
            photo_base64 = photos[0]
            img_data = base64.b64decode(photo_base64.split(',')[1])
            img = Image.open(io.BytesIO(img_data))
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            predicted_identity, confidence = predict_face(img_cv)
            missing_person.predicted_identity = predicted_identity
            missing_person.confidence = confidence
        
        db.session.add(missing_person)
        db.session.commit()

        report = Report(
            guardian_id=guardian.id,
            missing_person_id=missing_person.id
        )
        db.session.add(report)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Report submitted successfully',
            'report_id': report.id,
            'prediction': {'identity': predicted_identity, 'confidence': confidence}
        })
        
    except Exception as e:
        db.session.rollback()
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
        
        img_data = base64.b64decode(photo_base64.split(',')[1])
        img = Image.open(io.BytesIO(img_data))
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        predicted_class, confidence = predict_face(img_cv)
        
        matches = []
        if confidence > 0.7:
            persons = MissingPerson.query.filter_by(predicted_identity=predicted_class).all()
            for person in persons:
                report = Report.query.filter_by(missing_person_id=person.id).first()
                if report:
                    matches.append({
                        'report_id': report.id,
                        'missing_person': {
                            'name': person.name,
                            'age': person.age,
                            'last_seen_location': person.last_seen_location
                        },
                        'guardian': {
                            'name': report.guardian.name,
                            'phone': report.guardian.phone
                        },
                        'confidence': person.confidence,
                        'predicted_identity': person.predicted_identity
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
    reports = Report.query.all()
    results = []
    for report in reports:
        results.append({
            'id': report.id,
            'status': report.status,
            'timestamp': report.timestamp,
            'guardian': {
                'name': report.guardian.name,
                'phone': report.guardian.phone
            },
            'missing_person': {
                'name': report.missing_person.name,
                'age': report.missing_person.age,
                'last_seen_location': report.missing_person.last_seen_location
            }
        })
    return jsonify({'success': True, 'reports': results})

@app.route('/api/search_by_name', methods=['GET'])
def search_by_name():
    """Search missing persons by name"""
    name_query = request.args.get('name', '').lower()
    persons = MissingPerson.query.filter(MissingPerson.name.ilike(f'%{name_query}%')).all()
    results = []
    for person in persons:
        report = Report.query.filter_by(missing_person_id=person.id).first()
        if report:
            results.append({
                'report_id': report.id,
                'missing_person': {
                    'name': person.name,
                    'age': person.age,
                },
                'guardian': {
                    'name': report.guardian.name
                }
            })
    return jsonify({'success': True, 'results': results})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    
    print("Starting Returno Face Recognition Server...")
    print("Server running on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)