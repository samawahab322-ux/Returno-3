import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
import os
import base64
from PIL import Image
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
import io

app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

DATASET_DIR = 'dataset' 

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

def preprocess_image(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (200, 200))
    return gray

def compare_with_dataset(img_cv):
    input_img = preprocess_image(img_cv)
    best_match = None
    highest_score = 0

    for filename in os.listdir(DATASET_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(DATASET_DIR, filename)
            dataset_img = cv2.imread(path)
            dataset_img = preprocess_image(dataset_img)
            
            score = ssim(input_img, dataset_img)
            if score > highest_score:
                highest_score = score
                best_match = filename

    return best_match, highest_score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    reports_from_db = Report.query.all()
    reports_for_template = []
    for report in reports_from_db:
        reports_for_template.append({
            'id': report.id,
            'status': report.status.title(),
            'timestamp': report.timestamp.strftime('%Y-%m-%d %H:%M'),
            'guardian': {
                'name': report.guardian.name,
                'phone': report.guardian.phone,
                'email': 'Not Available', 
                'address': 'Not Available' 
            },
            'missing_person': {
                'name': report.missing_person.name,
                'age': report.missing_person.age,
                'last_seen_location': report.missing_person.last_seen_location,
                'description': report.missing_person.description,
                'last_seen_time': 'Not Available' 
            }
        })
    return render_template('dashboard.html', reports=reports_for_template)

@app.route('/api/report_missing', methods=['POST'])
def report_missing():
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
        db.session.add(missing_person)
        db.session.commit() 

        photos = data.get('photos', [])
        predicted_identity = None
        confidence = 0.0

        if photos:
            photo_base64 = photos[0]
            img_data = base64.b64decode(photo_base64.split(',')[1])
            img = Image.open(io.BytesIO(img_data))
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            upload_dir = 'uploads/reports'
            os.makedirs(upload_dir, exist_ok=True)
            image_filename = f"{missing_person.id}.jpg"
            image_path = os.path.join(upload_dir, image_filename)
            cv2.imwrite(image_path, img_cv)

            predicted_identity, confidence = compare_with_dataset(img_cv)
            missing_person.predicted_identity = predicted_identity
            missing_person.confidence = confidence
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
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/search_by_photo', methods=['POST'])
def search_by_photo():
    try:
        data = request.json
        photo_base64 = data.get('photo')
        img_data = base64.b64decode(photo_base64.split(',')[1])
        img = Image.open(io.BytesIO(img_data))
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        match_filename, score = compare_with_dataset(img_cv)
        matches = []

        if score > 0.7:
            person_name = os.path.splitext(match_filename)[0]
            persons = MissingPerson.query.filter(MissingPerson.name.ilike(f"%{person_name}%")).all()
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
                        'predicted_identity': person.predicted_identity,
                        'confidence': person.confidence
                    })

        return jsonify({
            'success': True,
            'predicted_identity': match_filename,
            'confidence': score,
            'matches': matches
        })

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/get_missing_reports', methods=['GET'])
def get_missing_reports():
    try:
        query = Report.query.join(MissingPerson)

        # Filter by Case ID
        case_id = request.args.get('case_id')
        if case_id:
            query = query.filter(Report.id == case_id)

        # Filter by Age Range
        min_age = request.args.get('min_age')
        max_age = request.args.get('max_age')
        if min_age and max_age:
            query = query.filter(MissingPerson.age.between(min_age, max_age))
        elif min_age:
            query = query.filter(MissingPerson.age >= min_age)
        elif max_age:
            query = query.filter(MissingPerson.age <= max_age)
            
        # Filter by Location
        location = request.args.get('location')
        if location:
            query = query.filter(MissingPerson.last_seen_location.ilike(f"%{location}%"))

        # Filter by Date Range
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        if start_date and end_date:
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
            query = query.filter(Report.timestamp.between(start_date_obj, end_date_obj))

        reports = query.all()
        results = []
        for report in reports:
            results.append({
                'id': report.id,
                'status': report.status,
                'timestamp': report.timestamp.strftime('%Y-%m-%d %H:%M'),
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
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/search_by_name', methods=['GET'])
def search_by_name():
    name_query = request.args.get('name', '').lower()
    persons = MissingPerson.query.filter(MissingPerson.name.ilike(f'%{name_query}%')).all()
    results = []
    for person in persons:
        report = Report.query.filter_by(missing_person_id=person.id).first()
        if report:
            results.append({
                'report_id': report.id,
                'missing_person': {'name': person.name, 'age': person.age},
                'guardian': {'name': report.guardian.name}
            })
    return jsonify({'success': True, 'results': results})


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    os.makedirs('uploads/reports', exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
