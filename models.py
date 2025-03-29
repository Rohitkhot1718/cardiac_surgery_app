from flask_login import UserMixin
from flask_mongoengine import MongoEngine
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = MongoEngine()

class User(UserMixin, db.Document):
    username = db.StringField(required=True, unique=True)
    email = db.StringField(required=True, unique=True)
    password_hash = db.StringField(required=True)
    role = db.StringField(required=True, choices=["doctor", "user"]) 
    last_login = db.DateTimeField()

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Patient(db.Document):
    user_id = db.ReferenceField(User, required=True)  
    name = db.StringField(required=True)
    age = db.IntField(required=True)
    gender = db.StringField(choices=["Male", "Female"], required=True)
    diabetes = db.IntField()
    hypertension = db.IntField()
    kidney_disease = db.IntField()
    respiratory_issues = db.IntField()
    ejection_fraction = db.IntField()
    surgery_type = db.StringField()
    risk_level = db.IntField(required=True)
    risk_status = db.StringField(required=True)
    risk_message = db.StringField(required=True)
    risk_factors = db.StringField(required=True)
    created_at = db.DateTimeField(default=datetime.utcnow)

