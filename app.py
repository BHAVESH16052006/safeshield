from flask import Flask, render_template, request, jsonify
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

try:
	# Load the model and vectorizer
	model = joblib.load('spam_model.pkl')
	vectorizer = joblib.load('vectorizer.pkl')
	logger.info("Model and vectorizer loaded successfully")
except Exception as e:
	logger.error(f"Error loading model or vectorizer: {str(e)}")

# Create directory for uploaded files if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
	os.makedirs(UPLOAD_FOLDER)
	logger.info(f"Created uploads directory at {UPLOAD_FOLDER}")

# Store prediction history
prediction_history = []

@app.route('/')
def home():
	try:
		return render_template('index.html')
	except Exception as e:
		logger.error(f"Error in home route: {str(e)}")
		return str(e), 500

@app.route('/predict', methods=['POST'])
def predict():
	if 'vectorizer' not in globals() or 'model' not in globals():
		return jsonify({'error': 'Model or vectorizer not loaded.'}), 500
	try:
		message = request.form.get('comment', '')
		file = request.files.get('email_file')
		attachments = []
		
		logger.info("Received prediction request")
		
		# Process uploaded file if present
		if file and file.filename:
			filename = os.path.join(UPLOAD_FOLDER, file.filename)
			file.save(filename)
			attachments.append(filename)
			logger.info(f"Saved uploaded file: {filename}")
			
			# Read email content from file
			with open(filename, 'r', encoding='utf-8') as f:
				message = f.read()
		
		# Get prediction
		message_vec = vectorizer.transform([message])
		prediction = model.predict(message_vec)[0]
		prediction_proba = model.predict_proba(message_vec)[0]
		confidence = max(prediction_proba) * 100
		
		# Store prediction in history
		prediction_history.append({
			'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
			'message': message[:100] + '...' if len(message) > 100 else message,
			'result': 'Spam' if prediction == 1 else 'Not Spam',
			'confidence': f'{confidence:.2f}%'
		})
		
		# Keep only last 50 predictions
		if len(prediction_history) > 50:
			prediction_history.pop(0)
		
		# Clean up attachments
		for attachment in attachments:
			if os.path.exists(attachment):
				os.remove(attachment)
		
		response = {
			'prediction': 'Spam' if prediction == 1 else 'Not Spam',
			'confidence': f'{confidence:.2f}%'
		}
		
		logger.info(f"Prediction completed successfully: {response}")
		return jsonify(response)
		
	except Exception as e:
		logger.error(f"Error in predict route: {str(e)}")
		return jsonify({'error': str(e)}), 500

@app.route('/analytics')
def get_analytics():
	# Calculate spam distribution
	spam_count = sum(1 for p in prediction_history if p['result'] == 'Spam')
	ham_count = len(prediction_history) - spam_count
	
	# Calculate accuracy trend (sample data for now)
	accuracy_trend = {
		'labels': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
		'data': [85, 88, 90, 92, 95, 97]
	}
	
	return jsonify({
		'spam_distribution': {
			'spam': spam_count,
			'ham': ham_count
		},
		'accuracy_trend': accuracy_trend,
		'prediction_history': prediction_history[-10:]  # Return last 10 predictions
	})

@app.route('/generate_report', methods=['POST'])
def generate_report():
	data = request.get_json()
	message = data.get('message', '')
	prediction = data.get('prediction', 'Unknown')
	confidence = data.get('confidence', 'Unknown')
	timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

	# Simple key indicators and recommendations
	key_indicators = []
	recommendations = []
	if prediction.lower() == 'spam':
		key_indicators.append('Message classified as spam')
		recommendations.append('Do not click on any links or download attachments.')
		recommendations.append('Mark as spam in your email client.')
	else:
		key_indicators.append('Message classified as not spam')
		recommendations.append('No action needed. This message appears safe.')

	report = {
		'timestamp': timestamp,
		'prediction': prediction,
		'confidence': confidence,
		'message_preview': message[:200] + ('...' if len(message) > 200 else ''),
		'analysis': {
			'key_indicators': key_indicators,
			'recommendations': recommendations
		}
	}
	return jsonify(report)

@app.route('/share_result', methods=['POST'])
def share_result():
	data = request.get_json()
	message = data.get('message', '')
	prediction = data.get('prediction', 'Unknown')
	confidence = data.get('confidence', 'Unknown')
	share_text = f"Prediction: {prediction}\nConfidence: {confidence}\nMessage Preview: {message[:200]}{'...' if len(message) > 200 else ''}"
	return jsonify({'share_text': share_text})

if __name__ == '__main__':
	try:
		app.run(debug=True)
	except Exception as e:
		logger.error(f"Error starting application: {str(e)}")