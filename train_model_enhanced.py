import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set NLTK data path to the current directory
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Download required NLTK data
try:
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('stopwords', download_dir=nltk_data_dir)
except Exception as e:
    logger.warning(f"NLTK download warning (not critical): {str(e)}")

def extract_email_features(texts):
    """Extract additional email-specific features."""
    features = []
    for text in texts:
        features.append({
            'length': len(text),
            'word_count': len(text.split()),
            'has_url': int(bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))),
            'has_number': int(bool(re.search(r'\d', text))),
            'caps_ratio': sum(1 for c in text if c.isupper()) / (len(text) + 1),
            'has_currency': int(bool(re.search(r'[$€£¥]', text))),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'urgency_words': sum(word in text.lower() for word in ['urgent', 'important', 'action', 'required', 'immediate']),
        })
    return pd.DataFrame(features)

class EnhancedSpamDetector:
    def __init__(self):
        self.tfidf = None
        self.xgb = XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.confidence_threshold = 0.70
        logger.info("Initializing stopwords...")
        self.stop_words = set(stopwords.words('english'))
        logger.info("Initialization complete")

    def preprocess_text(self, text):
        """Basic text preprocessing using NLTK."""
        try:
            # Convert to lowercase
            text = text.lower()
            logger.debug(f"Lowercase text: {text[:50]}...")
            
            # Tokenize
            tokens = word_tokenize(text)
            logger.debug(f"Tokenized count: {len(tokens)}")
            
            # Remove stopwords and non-alphabetic tokens
            tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
            logger.debug(f"Clean tokens count: {len(tokens)}")
            
            return ' '.join(tokens)
        except Exception as e:
            logger.error(f"Error in preprocess_text: {str(e)}")
            raise

    def extract_features(self, X):
        """Extract both TF-IDF and custom features."""
        try:
            logger.info("Starting feature extraction...")
            
            # Preprocess texts
            logger.info(f"Preprocessing {len(X)} texts...")
            processed_texts = [self.preprocess_text(text) for text in X]
            logger.info("Text preprocessing complete")
            
            # Get TF-IDF features
            logger.info("Extracting TF-IDF features...")
            if self.tfidf is None:
                logger.info("Initializing TF-IDF vectorizer...")
                self.tfidf = TfidfVectorizer(
                    max_features=5000,
                    min_df=2,
                    max_df=0.95,
                    ngram_range=(1, 3),
                    stop_words='english'
                )
                tfidf_features = self.tfidf.fit_transform(processed_texts)
                logger.info(f"TF-IDF features shape: {tfidf_features.shape}")
            else:
                tfidf_features = self.tfidf.transform(processed_texts)
            
            # Get custom features
            logger.info("Extracting custom features...")
            custom_features = extract_email_features(X)
            logger.info(f"Custom features shape: {custom_features.shape}")
            
            # Combine features
            logger.info("Combining features...")
            combined_features = np.hstack([
                tfidf_features.toarray(),
                custom_features.values
            ])
            logger.info(f"Final features shape: {combined_features.shape}")
            
            return combined_features
            
        except Exception as e:
            logger.error(f"Error in extract_features: {str(e)}")
            raise

    def fit(self, X, y):
        """Train the model."""
        try:
            logger.info("Starting model training...")
            logger.info(f"Input shapes - X: {X.shape}, y: {y.shape}")
            
            # Extract features
            X_features = self.extract_features(X)
            logger.info(f"Extracted features shape: {X_features.shape}")
            
            # Train XGBoost
            logger.info("Training XGBoost model...")
            self.xgb.fit(X_features, y)
            logger.info("XGBoost training complete")
            
            return self
            
        except Exception as e:
            logger.error(f"Error in fit method: {str(e)}")
            raise

    def predict(self, X):
        """Predict with confidence threshold."""
        try:
            logger.info("Starting prediction...")
            X_features = self.extract_features(X)
            probas = self.xgb.predict_proba(X_features)
            
            # Apply confidence threshold
            predictions = []
            for proba in probas:
                if max(proba) >= self.confidence_threshold:
                    predictions.append(1 if proba[1] > proba[0] else 0)
                else:
                    predictions.append(0)
            
            logger.info(f"Predictions complete. Shape: {len(predictions)}")
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error in predict method: {str(e)}")
            raise

    def predict_proba(self, X):
        """Get prediction probabilities."""
        try:
            logger.info("Starting probability prediction...")
            X_features = self.extract_features(X)
            probas = self.xgb.predict_proba(X_features)
            logger.info("Probability prediction complete")
            return probas
        except Exception as e:
            logger.error(f"Error in predict_proba method: {str(e)}")
            raise

def augment_training_data(df):
    """Add more legitimate business communications examples."""
    business_templates = [
        "Meeting scheduled for {time} to discuss {topic}. {regards}",
        "Please review the attached {document} for our {time} meeting. {regards}",
        "Following up on our discussion about {topic}. {regards}",
        "The {document} has been updated. Please review when you have a chance. {regards}",
        "Reminder: Team meeting at {time} in {location}. Agenda: {topic}. {regards}"
    ]
    
    times = ["2 PM", "3:30 PM", "11 AM", "next Tuesday", "tomorrow morning"]
    topics = ["project updates", "quarterly results", "budget proposal", "marketing strategy", "client feedback"]
    documents = ["report", "presentation", "documentation", "analysis", "proposal"]
    locations = ["Conference Room A", "Meeting Room 2", "Virtual Call", "Main Office", "Board Room"]
    regards = ["Best regards", "Thanks", "Regards", "Kind regards", "Sincerely"]
    
    new_legitimate_emails = []
    for template in business_templates:
        for _ in range(20):  # Generate 20 variations of each template
            email = template.format(
                time=np.random.choice(times),
                topic=np.random.choice(topics),
                document=np.random.choice(documents),
                location=np.random.choice(locations),
                regards=np.random.choice(regards)
            )
            new_legitimate_emails.append({"text": email, "label_num": 0})
    
    # Add new examples to the dataset
    augmented_df = pd.concat([
        df,
        pd.DataFrame(new_legitimate_emails)
    ], ignore_index=True)
    
    return augmented_df

def train_enhanced_model():
    try:
        # Load the dataset with explicit encoding
        logger.info("Loading dataset...")
        df = pd.read_csv("spam_ham_dataset.csv", encoding='utf-8', on_bad_lines='skip')
        df_data = df[["text", "label_num"]]
        
        # Clean the text data
        logger.info("Cleaning text data...")
        df_data['text'] = df_data['text'].astype(str).apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))
        
        # Augment training data with more legitimate business communications
        logger.info("Augmenting training data...")
        df_augmented = augment_training_data(df_data)

        # Split the data
        logger.info("Splitting dataset...")
        X_train, X_test, y_train, y_test = train_test_split(
            df_augmented['text'], 
            df_augmented['label_num'],
            test_size=0.2,
            random_state=42,
            stratify=df_augmented['label_num']
        )

        # Initialize and train the enhanced model
        logger.info("Training enhanced model...")
        model = EnhancedSpamDetector()
        model.fit(X_train, y_train)

        # Evaluate the model
        logger.info("Evaluating model...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model Accuracy: {accuracy:.4f}")
        
        # Print detailed classification report
        report = classification_report(y_test, y_pred)
        logger.info(f"Classification Report:\n{report}")

        # Save the model components separately
        logger.info("Saving model components...")
        model_data = {
            'model_class': EnhancedSpamDetector,
            'model_instance': model,
            'tfidf_vectorizer': model.tfidf,
            'xgboost_model': model.xgb,
            'stop_words': model.stop_words,
            'confidence_threshold': model.confidence_threshold
        }
        joblib.dump(model_data, 'spam_model_enhanced.pkl')
        
        logger.info("Enhanced model saved successfully!")
        return True

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        return False

if __name__ == "__main__":
    train_enhanced_model() 