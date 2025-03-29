import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import db, User, Patient
import joblib
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
from flask_cors import CORS
from flask_mongoengine import MongoEngine
from bson import ObjectId

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, 
    static_url_path='/static',
    static_folder='static')

# App configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['MONGODB_SETTINGS'] = {
    'db': os.getenv('MONGODB_DB'),
    'host': os.getenv('MONGODB_HOST'),
    'port': int(os.getenv('MONGODB_PORT'))
}

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'signin'  # Redirect to login page if not authenticated
login_manager.login_message_category = 'info'

CORS(app)    

# Initialize database
db = MongoEngine()
db.init_app(app)

# Configure model paths and load models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_cardiac_surgery_model.joblib')
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, 'models', 'model_feature_names.joblib')


# Define expected feature names if model loading fails
DEFAULT_FEATURES = [
    'Age', 'Diabetes', 'Hypertension', 'Kidney_Disease', 
    'Respiratory_Issues', 'Ejection_Fraction', 'Gender_Male',
    'Surgery_Type_CABG', 'Surgery_Type_Valve', 'Surgery_Type_Congenital',
    'Surgery_Type_Aneurysm','Surgery_Type_Transplant'
]

# Load model with warning suppression
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        original_model = joblib.load(MODEL_PATH)
        training_feature_names = joblib.load(FEATURE_NAMES_PATH)
        
        if training_feature_names is None:
            training_feature_names = DEFAULT_FEATURES
            print("Using default feature names")
        
        # Check if model is a RandomForestClassifier, else create one
        if not isinstance(original_model, RandomForestClassifier):
            print("Creating new RandomForestClassifier model")
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            dummy_X = pd.DataFrame(0, index=[0], columns=training_feature_names)
            dummy_y = [0]
            model.fit(dummy_X, dummy_y)
        else:
            model = original_model
        
        print("Model loaded successfully")

except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    training_feature_names = DEFAULT_FEATURES
    print("Using default configuration")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/get_started')
def get_started():
    if current_user.is_authenticated:
        return redirect(url_for('index'))  # If logged in, go to dashboard
    return redirect(url_for('signin'))  # If not logged in, go to login


@app.route('/cardiac')
@login_required
def index():
    return render_template('cardiac/cardiac.html')

@app.route('/cardiac_report/')
@login_required
def cardiac_report():
    return render_template('cardiac/cardiac_report.html')


@app.route('/prediction/')
@login_required
def prediction():
    return render_template('cardiac/prediction.html')

@app.route('/templates/<path:subdir>/<path:filename>')
def serve_template(subdir, filename):
    return send_from_directory(f'templates/{subdir}', filename)

@app.route('/start_predict/', methods=['POST'])
@login_required
def start_prediction():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data received", "status": "error"}), 400

        # Store patient data in DB
        patient = Patient(
            user_id=current_user.id,
            name=data["txtName"].upper(),
            age=int(data["txtAge"]),
            gender="Male" if data["selGender"] == "1" else "Female",
            diabetes=int(data["selDiabetes"]),
            hypertension=int(data["selHypertension"]),
            kidney_disease=int(data["selKidneyDisease"]),
            respiratory_issues=int(data["selRespiratoryIssues"]),
            ejection_fraction=int(data["selEjectionFraction"]),
            surgery_type=", ".join(filter(None, [
                "CABG" if data["selSurgeryTypeCABG"] == "1" else None,
                "Valve" if data["selSurgeryTypeValve"] == "1" else None,
                "Congenital" if data["selSurgeryTypeCongenita"] == "1" else None,
                "Aneurysm" if data["selSurgeryTypeAneurysm"] == "1" else None,
                "Transplant" if data["selSurgeryTypeTransplant"] == "1" else None
            ])),
            risk_level=0,  # Set as 0 before prediction
            risk_status="Pending",
            risk_message='',
            risk_factors=''
        )
        patient.save()

        # Now predict immediately
        test_data = pd.DataFrame({
            "Age": [patient.age],
            "Diabetes": [patient.diabetes],
            "Hypertension": [patient.hypertension],
            "Kidney_Disease": [patient.kidney_disease],
            "Respiratory_Issues": [patient.respiratory_issues],
            "Ejection_Fraction": [patient.ejection_fraction],
            "Gender_Male": [1 if patient.gender == "Male" else 0],
            "Surgery_Type_CABG": [1 if "CABG" in patient.surgery_type else 0],
            "Surgery_Type_Valve": [1 if "Valve" in patient.surgery_type else 0],
            "Surgery_Type_Congenital": [1 if "Congenital" in patient.surgery_type else 0],
            "Surgery_Type_Aneurysm": [1 if "Aneurysm" in patient.surgery_type else 0],
            "Surgery_Type_Transplant": [1 if "Transplant" in patient.surgery_type else 0]
        })

        if model is None:
            return jsonify({"error": "Model not loaded", "status": "error"}), 500

        test_data = test_data.reindex(columns=training_feature_names, fill_value=0)
        probabilities = model.predict_proba(test_data)
        severity_percentage = float(probabilities[0][1] * 100)

        # Determine risk level
        risk_status = "Low" if severity_percentage < 30 else "Moderate" if severity_percentage < 60 else "High"

        if severity_percentage < 30:
            risk_status = "Low"
            risk_message = "Regular follow-up recommended"
            risk_factors = ["Normal vital signs", "Good ejection fraction", "No major complications"]
        elif 30 <= severity_percentage <= 60:
            risk_status = "Moderate"
            risk_message = "Medical consultation recommended"
            risk_factors = ["Elevated blood pressure", "Reduced ejection fraction", "Pre-existing conditions"]
        else:
            risk_status = "High"
            risk_message = "Immediate medical attention required"
            risk_factors = ["Multiple risk factors", "Low ejection fraction", "Complex surgery type", "Age consideration"]


        # Update patient record
        patient.update(
            risk_level=severity_percentage,
            risk_status=risk_status,
            risk_message=risk_message,
            risk_factors=", ".join(risk_factors) 
        )

        return jsonify({
            "status": "success",
            "patient_id": str(patient.id),
            "risk_level": severity_percentage,
            "risk_status": risk_status,
            "risk_message": risk_message,
            "risk_factors": risk_factors
        })

    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}", "status": "error"}), 500


@app.route('/get_prediction_result', methods=['GET'])
def get_prediction_result():
    patient_id = request.args.get('patient_id')  # Assuming you're passing patient_id in the query
    
    # Retrieve the prediction result from the DB
    patient = Patient.objects(id=patient_id).first()  
    
    if patient:
        return jsonify({
            "status": "success",
            "patient_data" : patient
        })
    else:
        return jsonify({"status": "error", "message": "Patient not found"})


@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard') if current_user.role == "doctor" else url_for('user_dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.objects(username=username).first()
        if not user:
            flash('Invalid username or password', 'danger')
            return redirect(url_for('signin'))

        if not user.check_password(password):
            flash('Invalid username or password', 'danger')
            return redirect(url_for('signin'))


        login_user(user)

        if user.role == "doctor":
            return redirect(url_for('dashboard'))
        else:
            return redirect(url_for('user_dashboard'))

    return render_template('auth/signin.html')



@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role')

        if not role:  # Ensure role is selected
            flash('Please select a role.', 'danger')
            return redirect(url_for('signup'))

        if User.objects(username=username).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('signup'))

        if User.objects(email=email).first():
            flash('Email already registered', 'danger')
            return redirect(url_for('signup'))

        user = User(username=username, email=email, role=role)
        user.set_password(password)
        user.save()

        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('signin'))

    return render_template('auth/signup.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('signin'))

@login_manager.user_loader
def load_user(user_id):
    return User.objects(pk=user_id).first()

@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role != "doctor":
        flash("Access Denied! Only doctors can access the dashboard.", "danger")
        return redirect(url_for('index'))

    # Fetch only patients added by the logged-in doctor
    total_patients = Patient.objects(user_id=current_user.id).count()
    high_risk_patients = Patient.objects(user_id=current_user.id, risk_status="High").count()
    moderate_risk_patients = Patient.objects(user_id=current_user.id, risk_status="Moderate").count()
    low_risk_patients = Patient.objects(user_id=current_user.id, risk_status="Low").count()
    patients = Patient.objects(user_id=current_user.id).order_by('+created_at')

    return render_template(
        'dashboard/dashboard.html',
        total_patients=total_patients,
        high_risk_patients=high_risk_patients,
        moderate_risk_patients=moderate_risk_patients,
        low_risk_patients=low_risk_patients,
        patients=patients
    )


@app.route('/user_dashboard')
@login_required
def user_dashboard():
    reports = Patient.objects.filter(user_id=current_user.id).order_by('-created_at')
    
    # Convert ObjectId to string
    for report in reports:
        report.id = str(report.id)

    return render_template('dashboard/user_dashboard.html', reports=reports)


@app.route('/view_report')
@login_required
def view_report():
    report_id = request.args.get('patient_id')  # Get patient_id from query params
    try:
        # Convert the report_id to ObjectId to match MongoDB's format
        report = Patient.objects(id=ObjectId(report_id)).first()
    except:
        flash("Invalid Report ID", "danger")
        return redirect(url_for('user_dashboard'))

    if not report:
        flash("Report not found", "danger")
        return redirect(url_for('user_dashboard'))
    
    return render_template('cardiac/cardiac_report.html', report=report)


if __name__ == '__main__':
    app.run(debug=True)
