# Predictive Model for Post-Operative Complications in Cardiac Surgery

A machine learning-based web application that helps healthcare professionals predict and assess post-operative complications in cardiac surgery patients.

## 🏥 Project Overview

This application uses a Random Forest Classifier to predict the risk level of post-operative complications based on patient data and surgery type. It provides real-time risk assessment and generates detailed reports for healthcare providers.

## 📋 Prerequisites

- Python 3.10
- MongoDB 6.0 or higher

## 🚀 Features

### Risk Assessment
- Real-time prediction of post-operative complications
- Risk level classification (Low, Moderate, High)
- Customized risk factor analysis
- Automated report generation

### User Management
- Role-based access (Doctors and Patients)
- Secure authentication system
- Personalized dashboards

### Reporting System
- Detailed PDF reports
- Patient history tracking
- Risk factor visualization
- Surgery type analysis

## 💻 Tech Stack

- **Python:** 3.10
- **Backend:** Python Flask 2.2.5
- **Database:** MongoDB
- **Authentication:** Flask-Login
- **ML Model:** Scikit-learn (Random Forest Classifier)
- **Frontend:** HTML, CSS, JavaScript
- **Reports:** html2pdf.js

## ⚙️ Installation

1. **Ensure Python 3.10 is installed:**
```bash
python --version
```

2. **Clone the repository:**
```bash
git clone <repository-url>
cd cardiac_surgery_app
```

3. **Set up virtual environment:**
```bash
python -m venv venv --python=python3.10
venv\Scripts\activate
```

4. **Upgrade pip and install dependencies:**
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. **Create .env file:**
```env
SECRET_KEY=your_secret_key
MONGODB_DB=cardiac_surgery_db
MONGODB_HOST=localhost
MONGODB_PORT=27017
```

6. **Run the application:**
```bash
python app.py
```

## 🔍 Features in Detail

### Risk Assessment Parameters
- Age
- Diabetes Status
- Hypertension
- Kidney Disease
- Respiratory Issues
- Ejection Fraction
- Gender
- Surgery Types:
  - CABG
  - Valve
  - Congenital
  - Aneurysm
  - Transplant

### Risk Classification
- **Low Risk** (<30%): Regular follow-up
- **Moderate Risk** (30-60%): Medical consultation
- **High Risk** (>60%): Immediate attention

## 👥 User Roles

### Doctor
- Add patient records
- View risk assessments
- Generate reports
- Track patient history

### Patient
- View personal reports
- Download PDF reports
- Track risk status

## 📝 Development Notes

- Python 3.10 is required for compatibility with machine learning libraries
- Virtual environment usage is strongly recommended
- MongoDB must be running before starting the application


## 📋 License

This project is licensed under the MIT License.