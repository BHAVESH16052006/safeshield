import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and vectorizer
model = joblib.load('spam_model_advanced.pkl')
vectorizer = joblib.load('vectorizer_advanced.pkl')

# Test messages
messages = [
    "Hi Sarah, Just wanted to confirm our meeting tomorrow at 2 PM to discuss the project updates. Best regards, John",
    "Dear team, Please find attached the quarterly report for your review. Thanks, Mike",
    "Hello, Following up on our discussion about the budget proposal. When would be a good time to meet? Regards, Lisa",
    "Meeting reminder: Project status update at 3 PM in Conference Room B. Agenda attached."
]

for message in messages:
    # Transform the message
    message_vec = vectorizer.transform([message])

    # Get prediction
    prediction = model.predict(message_vec)[0]
    prediction_proba = model.predict_proba(message_vec)[0]
    confidence = max(prediction_proba) * 100

    print("\nMessage:", message)
    print(f"Prediction: {'Spam' if prediction == 1 else 'Not Spam'}")
    print(f"Confidence: {confidence:.2f}%") 