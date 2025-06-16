from flask import Flask, render_template, request, jsonify, session
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import joblib
import os
from datetime import datetime
import logging
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from train_model_enhanced import EnhancedSpamDetector
import imaplib
import email
from email.header import decode_header
import PyPDF2
import docx
import threading
import queue
from cryptography.fernet import Fernet
import base64

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set NLTK data path to the current directory
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
	os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Encryption key for email credentials
ENCRYPTION_KEY = Fernet.generate_key()
cipher_suite = Fernet(ENCRYPTION_KEY)

# Email processing queue
email_queue = queue.Queue()
processing_status = {'total': 0, 'processed': 0, 'results': []}

try:
	# Load the model components
	logger.info("Loading model components...")
	model_data = joblib.load('spam_model_enhanced.pkl')
	
	# Create a new instance of the model
	model = model_data['model_class']()
	model.tfidf = model_data['tfidf_vectorizer']
	model.xgb = model_data['xgboost_model']
	model.stop_words = model_data['stop_words']
	model.confidence_threshold = model_data['confidence_threshold']
	
	logger.info("Enhanced model loaded successfully")
except Exception as e:
	logger.error(f"Error loading enhanced model: {str(e)}")
	model = None

# Create directory for uploaded files if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
	os.makedirs(UPLOAD_FOLDER)
	logger.info(f"Created uploads directory at {UPLOAD_FOLDER}")

# Store prediction history
prediction_history = []

def extract_text_from_document(file_path):
	"""Extract text from various document formats."""
	logger.info(f"Extracting text from: {file_path}")
	text = ""
	file_ext = os.path.splitext(file_path)[1].lower()
	
	try:
		if file_ext == '.pdf':
			logger.info("Processing PDF file...")
			with open(file_path, 'rb') as file:
				pdf_reader = PyPDF2.PdfReader(file)
				for page in pdf_reader.pages:
					text += page.extract_text() + "\n"
		
		elif file_ext == '.docx':
			logger.info("Processing DOCX file...")
			doc = docx.Document(file_path)
			for para in doc.paragraphs:
				text += para.text + "\n"
		
		elif file_ext in ['.txt', '.csv', '.md']:
			logger.info("Processing text file...")
			with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
				text = file.read()
		
		text = text.strip()
		logger.info(f"Successfully extracted {len(text)} characters")
		return text
	except Exception as e:
		logger.error(f"Error extracting text from document: {str(e)}")
		return None

def analyze_document_content(text):
	"""Analyze document content and generate detailed report."""
	try:
		logger.info("Starting document analysis...")
		
		# Preprocess text
		processed_text = text.lower()
		words = processed_text.split()
		
		# Extract features
		features = {
			'length': len(text),
			'word_count': len(words),
			'urls': len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
			'caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
			'exclamation_marks': text.count('!'),
			'question_marks': text.count('?'),
			'numbers': len(re.findall(r'\d+', text)),
			'suspicious_words': len(re.findall(r'\b(urgent|important|money|bank|account|password|verify|click|win|prize)\b', text.lower()))
		}
		
		logger.info("Features extracted successfully")
		
		# Get prediction
		prediction = model.predict([text])[0]
		probas = model.predict_proba([text])[0]
		confidence = max(probas) * 100
		
		logger.info(f"Prediction complete - Spam: {prediction}, Confidence: {confidence:.2f}%")
		
		# Generate detailed analysis
		suspicious_patterns = {
			'urgency': len(re.findall(r'\b(urgent|immediate|act now|limited time)\b', text.lower())),
			'money': len(re.findall(r'\b(money|cash|dollar|prize|winner)\b', text.lower())),
			'sensitive_info': len(re.findall(r'\b(password|account|credit card|ssn|social security)\b', text.lower())),
			'pressure': len(re.findall(r'\b(must|need|expire|terminate|last chance)\b', text.lower()))
		}
		
		return {
			'is_spam': prediction == 1,
			'confidence': confidence,
			'features': features,
			'suspicious_patterns': suspicious_patterns,
			'text_preview': text[:500] + '...' if len(text) > 500 else text,
			'analysis_details': {
				'caps_percentage': features['caps_ratio'] * 100,
				'url_count': features['urls'],
				'suspicious_word_count': features['suspicious_words'],
				'urgency_indicators': suspicious_patterns['urgency'],
				'pressure_tactics': suspicious_patterns['pressure']
			}
		}
		
	except Exception as e:
		logger.error(f"Error in document analysis: {str(e)}")
		return None

def process_email_batch(email_obj, password):
	"""Process emails in batches."""
	try:
		# Connect to Gmail IMAP server
		mail = imaplib.IMAP4_SSL("imap.gmail.com")
		mail.login(email_obj, password)
		mail.select("inbox")
		
		# Search for all emails
		_, messages = mail.search(None, "ALL")
		email_ids = messages[0].split()
		
		# Update total count
		processing_status['total'] = len(email_ids)
		processing_status['processed'] = 0
		processing_status['results'] = []
		
		# Process each email
		for email_id in email_ids:
			_, msg = mail.fetch(email_id, "(RFC822)")
			email_body = msg[0][1]
			email_message = email.message_from_bytes(email_body)
			
			# Extract subject and body
			subject = decode_header(email_message["subject"])[0][0]
			if isinstance(subject, bytes):
				subject = subject.decode()
			
			body = ""
			if email_message.is_multipart():
				for part in email_message.walk():
					if part.get_content_type() == "text/plain":
						body += part.get_payload(decode=True).decode()
			else:
				body = email_message.get_payload(decode=True).decode()
			
			# Analyze content
			content = f"Subject: {subject}\n\n{body}"
			result = analyze_document_content(content)
			
			if result:
				result.update({
					'subject': subject,
					'date': email_message["date"],
					'from': email_message["from"]
				})
				processing_status['results'].append(result)
			
			processing_status['processed'] += 1
		
		mail.logout()
		
	except Exception as e:
		logger.error(f"Error processing emails: {str(e)}")
		processing_status['error'] = str(e)

@app.route('/')
def home():
	try:
		if model is None:
			return render_template('index.html', error="Model not loaded. Please check server logs.")
		return render_template('index.html')
	except Exception as e:
		logger.error(f"Error in home route: {str(e)}")
		return str(e), 500

@app.route('/analyze_document', methods=['POST'])
def analyze_document():
	try:
		# Create uploads directory if it doesn't exist
		if not os.path.exists(UPLOAD_FOLDER):
			os.makedirs(UPLOAD_FOLDER)
			logger.info(f"Created uploads directory: {UPLOAD_FOLDER}")

		file = request.files.get('document')
		logger.info(f"Received file: {file.filename if file else 'No file'}")
		
		if not file:
			logger.error("No document provided")
			return jsonify({'error': 'No document provided'}), 400
		
		# Validate file type
		allowed_extensions = {'.txt', '.pdf', '.docx'}
		file_ext = os.path.splitext(file.filename)[1].lower()
		if file_ext not in allowed_extensions:
			logger.error(f"Invalid file type: {file_ext}")
			return jsonify({'error': f'Invalid file type. Allowed types: {", ".join(allowed_extensions)}'}), 400
		
		# Save file with secure filename
		filename = os.path.join(UPLOAD_FOLDER, file.filename)
		logger.info(f"Saving file to: {filename}")
		file.save(filename)
		
		# Extract text
		logger.info("Extracting text from document...")
		text = extract_text_from_document(filename)
		if not text:
			logger.error("Could not extract text from document")
			return jsonify({'error': 'Could not extract text from document'}), 400
		
		logger.info(f"Extracted text length: {len(text)}")
		
		# Analyze content
		logger.info("Analyzing document content...")
		result = analyze_document_content(text)
		
		# Clean up
		try:
			os.remove(filename)
			logger.info(f"Cleaned up temporary file: {filename}")
		except Exception as e:
			logger.warning(f"Could not remove temporary file: {str(e)}")
		
		if result:
			logger.info(f"Analysis complete. Result: {result['is_spam']}, Confidence: {result['confidence']:.2f}%")
			return jsonify(result)
		else:
			logger.error("Analysis failed")
			return jsonify({'error': 'Analysis failed'}), 500
		
	except Exception as e:
		logger.error(f"Error analyzing document: {str(e)}")
		return jsonify({'error': str(e)}), 500

@app.route('/start_email_analysis', methods=['POST'])
def start_email_analysis():
	try:
		data = request.get_json()
		email_address = data.get('email')
		password = data.get('password')
		
		if not email_address or not password:
			return jsonify({'error': 'Email and password required'}), 400
		
		# Store encrypted credentials in session
		session['email'] = base64.b64encode(cipher_suite.encrypt(email_address.encode())).decode()
		session['password'] = base64.b64encode(cipher_suite.encrypt(password.encode())).decode()
		
		# Start processing in background
		threading.Thread(target=process_email_batch, args=(email_address, password)).start()
		
		return jsonify({'message': 'Email analysis started'})
		
	except Exception as e:
		logger.error(f"Error starting email analysis: {str(e)}")
		return jsonify({'error': str(e)}), 500

@app.route('/email_analysis_status')
def email_analysis_status():
	try:
		return jsonify({
			'total': processing_status['total'],
			'processed': processing_status['processed'],
			'results': processing_status['results'],
			'error': processing_status.get('error')
		})
	except Exception as e:
		logger.error(f"Error getting email analysis status: {str(e)}")
		return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
	if model is None:
		return jsonify({'error': 'Model not loaded.'}), 500
	
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
		
		# Get prediction using enhanced model
		prediction = model.predict([message])[0]
		prediction_proba = model.predict_proba([message])[0]
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
			'confidence': f'{confidence:.2f}%',
			'features': {
				'length': len(message),
				'has_url': 'http' in message.lower() or 'www' in message.lower(),
				'caps_ratio': sum(1 for c in message if c.isupper()) / len(message) if message else 0,
				'exclamation_count': message.count('!'),
				'question_count': message.count('?')
			}
		}
		
		logger.info(f"Prediction completed successfully: {response}")
		return jsonify(response)
		
	except Exception as e:
		logger.error(f"Error in predict route: {str(e)}")
		return jsonify({'error': str(e)}), 500

@app.route('/analytics')
def get_analytics():
	try:
		# Calculate spam distribution
		spam_count = sum(1 for p in prediction_history if p['result'] == 'Spam')
		ham_count = len(prediction_history) - spam_count
		
		# Calculate confidence distribution
		confidence_levels = {
			'high': sum(1 for p in prediction_history if float(p['confidence'].strip('%')) >= 90),
			'medium': sum(1 for p in prediction_history if 70 <= float(p['confidence'].strip('%')) < 90),
			'low': sum(1 for p in prediction_history if float(p['confidence'].strip('%')) < 70)
		}
		
		return jsonify({
			'spam_distribution': {
				'spam': spam_count,
				'ham': ham_count
			},
			'confidence_distribution': confidence_levels,
			'prediction_history': prediction_history[-10:]  # Return last 10 predictions
		})
	except Exception as e:
		logger.error(f"Error in analytics route: {str(e)}")
		return jsonify({'error': str(e)}), 500

@app.route('/generate_report', methods=['POST'])
def generate_report():
	try:
		data = request.get_json()
		message = data.get('message', '')
		prediction = data.get('prediction', 'Unknown')
		confidence = data.get('confidence', 'Unknown')
		timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

		# Generate key indicators
		key_indicators = []
		recommendations = []
		
		# Basic indicators
		if prediction.lower() == 'spam':
			key_indicators.extend([
				'Message classified as spam',
				f'Confidence level: {confidence}',
				f'Message length: {len(message)} characters',
				f'Contains URLs: {"Yes" if "http" in message.lower() or "www" in message.lower() else "No"}',
				f'Exclamation marks: {message.count("!")}',
				f'ALL CAPS ratio: {(sum(1 for c in message if c.isupper()) / len(message) if message else 0):.2%}'
			])
			recommendations.extend([
				'Do not click on any links or download attachments',
				'Mark as spam in your email client',
				'Consider adding sender to block list',
				'Report to IT security if this is a work email'
			])
		else:
			key_indicators.extend([
				'Message classified as legitimate',
				f'Confidence level: {confidence}',
				'No suspicious patterns detected'
			])
			recommendations.append('Message appears safe to interact with')

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
	except Exception as e:
		logger.error(f"Error generating report: {str(e)}")
		return jsonify({'error': str(e)}), 500

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