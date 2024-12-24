from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import PasswordField, SubmitField, EmailField
from wtforms.validators import InputRequired, Length, Email
from flask_bcrypt import Bcrypt
import os
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import random
import string
from math import radians, sin, cos, sqrt, atan2
import shutil
import re
import requests
import pandas as pd
# --------------------- Initialization ---------------------
# Define paths
PREDICTION_API_URL = 'http://localhost:5001/predict'  # URL of the Prediction API
TEMP_DIR = 'temp_processing'

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, 'database.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'thisisasecretkey')  # Ensure to set this in your environment

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# --------------------- User Model ---------------------

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False, unique=True)  # Email as username
    password = db.Column(db.String(80), nullable=False)
    code = db.Column(db.String(6), nullable=False)

# Create database tables
with app.app_context():
    logging.info("Creating all database tables...")
    db.create_all()
    logging.info("All tables created.")

# --------------------- Forms ---------------------

class RegisterForm(FlaskForm):
    username = EmailField(validators=[
        InputRequired(),
        Email(message='Invalid email'),
        Length(min=4, max=100)
    ], render_kw={"placeholder": "Email"})
    password = PasswordField(validators=[
        InputRequired(),
        Length(min=4, max=20)
    ], render_kw={"placeholder": "Password"})
    submit = SubmitField("Register")

class LoginForm(FlaskForm):
    username = EmailField(validators=[
        InputRequired(),
        Email(message='Invalid email'),
        Length(min=4, max=100)
    ], render_kw={"placeholder": "Email"})
    password = PasswordField(validators=[
        InputRequired(),
        Length(min=4, max=20)
    ], render_kw={"placeholder": "Password"})
    submit = SubmitField("Login")

class CodeForm(FlaskForm):
    code = PasswordField(validators=[
        InputRequired(),
        Length(min=6, max=6)
    ], render_kw={"placeholder": "Authentication Code"})
    submit = SubmitField("Submit")

class ConfirmForm(FlaskForm):
    username = EmailField(validators=[
        InputRequired(),
        Email(message='Invalid email'),
        Length(min=4, max=100)
    ], render_kw={"placeholder": "Email"})
    submit = SubmitField("Confirm")

class ResetForm(FlaskForm):
    password = PasswordField(validators=[
        InputRequired(),
        Length(min=4, max=20)
    ], render_kw={"placeholder": "New Password"})
    submit = SubmitField("Confirm")

# --------------------- Login Manager ---------------------

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --------------------- Email Function ---------------------

def send_authentication_email(username, code):
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    email_user = 'hieuminh23405@gmail.com'  # Replace with your email
    email_password = 'ztls pfjz xddh xuup'

    if not email_user or not email_password:
        logging.error("Email credentials are not set in environment variables.")
        raise Exception("Email credentials not configured.")

    try:
        smtp = smtplib.SMTP(smtp_server, smtp_port)
        smtp.ehlo()
        smtp.starttls()
        smtp.login(email_user, email_password)

        msg = MIMEMultipart()
        msg['Subject'] = "Authentication Code"
        msg['From'] = email_user
        msg['To'] = username
        msg.attach(MIMEText(f"This is your authentication code: {code}", 'plain'))

        smtp.sendmail(from_addr=email_user, to_addrs=username, msg=msg.as_string())
        smtp.quit()
        logging.info(f"Authentication email sent to {username}")
    except Exception as e:
        logging.error(f"Failed to send email to {username}: {str(e)}")
        raise

# --------------------- Code Generation ---------------------

def generate_and_store_code():
    code = ''.join(random.choices(string.digits, k=6))
    hashed_code = bcrypt.generate_password_hash(code).decode('utf-8')
    session['code'] = hashed_code
    return code

# --------------------- Routes ---------------------

# Login Route
@app.route('/', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            logging.info(f"User {user.username} logged in successfully.")
            flash('Logged in successfully.', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'danger')
            logging.warning(f"Failed login attempt for user {form.username.data}.")
    return render_template('login.html', form=form)

# Register Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(username=form.username.data).first()
        if existing_user:
            flash('That email already exists, choose a different email', 'danger')
            logging.warning(f"Registration attempt with existing email: {form.username.data}")
        else:
            code = generate_and_store_code()
            session['username'] = form.username.data
            session['password'] = form.password.data
            try:
                send_authentication_email(form.username.data, code)
                flash('An authentication code has been sent to your email.', 'info')
                logging.info(f"Registration initiated for {form.username.data}. Authentication code sent.")
                return redirect(url_for('code_reg'))
            except Exception as e:
                flash('Failed to send authentication email. Please try again.', 'danger')
                logging.error(f"Error sending authentication email: {str(e)}")
    return render_template('register.html', form=form)

# Authentication Code Route for Registration
@app.route('/code_reg', methods=['GET', 'POST'])
def code_reg():
    form = CodeForm()
    if form.validate_on_submit():
        entered_code = form.code.data
        stored_code = session.get('code')
        if stored_code and bcrypt.check_password_hash(stored_code, entered_code):
            session.pop('code')
            hashed_password = bcrypt.generate_password_hash(session['password']).decode('utf-8')
            new_user = User(username=session['username'], password=hashed_password, code="000000")
            db.session.add(new_user)
            db.session.commit()
            logging.info(f"User {new_user.username} registered successfully.")
            session.pop('username')
            session.pop('password')
            flash('Registration successful. You can now log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Invalid code. Please try again.', 'danger')
            logging.warning(f"Invalid authentication code entered: {entered_code}")
    return render_template('code_reg.html', form=form)

# Confirm Route
@app.route('/confirm', methods=['GET', 'POST'])
def confirm():
    form = ConfirmForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            code = generate_and_store_code()
            session['username'] = form.username.data
            try:
                send_authentication_email(form.username.data, code)
                flash('An authentication code has been sent to your email.', 'info')
                logging.info(f"Confirmation code sent to {form.username.data}")
                return redirect(url_for('check_code'))
            except Exception as e:
                flash('Failed to send authentication email. Please try again.', 'danger')
                logging.error(f"Error sending confirmation email: {str(e)}")
        else:
            flash('Email does not exist in our records. Please enter a valid email.', 'danger')
            logging.warning(f"Confirmation attempt with non-existent email: {form.username.data}")
    return render_template('confirm.html', form=form)

# Check Code Route for Confirmation
@app.route('/code', methods=['GET', 'POST'])
def check_code():
    form = CodeForm()
    if form.validate_on_submit():
        entered_code = form.code.data
        stored_code = session.get('code')
        if stored_code and bcrypt.check_password_hash(stored_code, entered_code):
            session.pop('code')
            return redirect(url_for('reset'))
        else:
            flash('Invalid code. Please try again.', 'danger')
            logging.warning(f"Invalid confirmation code entered: {entered_code}")
    return render_template('code.html', form=form)

# Reset Password Route
@app.route('/reset', methods=['GET', 'POST'])
def reset():
    form = ResetForm()
    username = session.get('username')
    if not username:
        flash('Session expired. Please confirm your email again.', 'warning')
        return redirect(url_for('confirm'))
    user = User.query.filter_by(username=username).first()
    if not user:
        flash('User does not exist. Please register again.', 'danger')
        return redirect(url_for('register'))

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user.password = hashed_password
        db.session.commit()
        session.pop('username')
        flash('Password reset successful. You can now log in with your new password.', 'success')
        logging.info(f"User {user.username} has reset their password.")
        return redirect(url_for('login'))
        
    return render_template('reset.html', form=form)

# Logout Route
@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    logging.info(f"User logged out.")
    return redirect(url_for('login'))

# Home Route (Protected)
@app.route('/home')
@login_required
def home():
    return render_template('home.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_type = request.form.get('input_type')
        
        if input_type == 'url':
            # Handle YouTube URL input
            audio_url = request.form.get('audio_url')
            if not audio_url:
                return jsonify({'status': 'error', 'message': 'No audio URL provided'}), 400
            os.environ['AUDIO_INPUT_TYPE'] = 'url'
            os.environ['AUDIO_INPUT_VALUE'] = audio_url
            
        elif input_type == 'record':
            # Handle the recorded audio blob
            if 'audio_data' not in request.files:
                return jsonify({'status': 'error', 'message': 'No audio data provided'}), 400
            
            audio_file = request.files['audio_data']
            try:
                temp_path = save_audio_blob(audio_file.read())
                os.environ['AUDIO_INPUT_TYPE'] = 'record'
                os.environ['AUDIO_INPUT_VALUE'] = temp_path
            except Exception as e:
                return jsonify({'status': 'error', 'message': f'Error saving audio: {str(e)}'}), 500
            
        else:
            return jsonify({'status': 'error', 'message': 'Invalid input type'}), 400

        # Process the audio data
        try:
            from data_process import main as process_main
            process_main()
        except Exception as e:
            logging.error(f"Error in data processing: {str(e)}")
            return jsonify({'status': 'error', 'message': f'Data processing error: {str(e)}'}), 500

        # Load the processed data and make prediction via Prediction API
        try:
            data = pd.read_csv('testing.csv')  # Ensure 'testing.csv' is generated correctly
            
            # Convert DataFrame to JSON
            input_json = data.to_dict(orient='records')[0]  # Assuming single record
            
            # Send POST request to Prediction API
            response = requests.post(PREDICTION_API_URL, json=input_json)
            
            # Log the response for debugging
            logging.info(f"Prediction API response status: {response.status_code}")
            logging.info(f"Prediction API response data: {response.json()}")
            
            if response.status_code == 200:
                prediction = response.json()
                return jsonify({
                    'status': 'success',
                    'prediction': prediction.get('prediction'),
                    'confidence': prediction.get('confidence')
                }), 200
            else:
                error_msg = response.json().get('message', 'Unknown error from Prediction API')
                return jsonify({'status': 'error', 'message': f'Prediction API error: {error_msg}'}), 500
            
        except Exception as e:
            logging.error(f"Error in prediction request: {str(e)}")
            return jsonify({'status': 'error', 'message': f'Prediction request error: {str(e)}'}), 500

    except Exception as e:
        logging.error(f"General error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    
    finally:
        # Clean up temporary files
        if os.path.exists(TEMP_DIR):
            import shutil
            shutil.rmtree(TEMP_DIR)

# --------------------- Helper Functions ---------------------

def save_audio_blob(audio_data):
    """Save the recorded audio blob to a WAV file."""
    TEMP_DIR = 'temp_processing'
    os.makedirs(TEMP_DIR, exist_ok=True)
    temp_path = os.path.join(TEMP_DIR, 'audio.wav')
    
    try:
        with open(temp_path, 'wb') as f:
            f.write(audio_data)
        logging.info(f"Audio data saved to {temp_path}")
        return temp_path
    except Exception as e:
        logging.error(f"Error saving audio blob: {str(e)}")
        raise

# --------------------- Run the App ---------------------

if __name__ == '__main__':
    app.run(debug=True)
