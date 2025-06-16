import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_advanced_model():
    try:
        # Load the dataset
        logger.info("Loading dataset...")
        df = pd.read_csv("spam_ham_dataset.csv")
        df_data = df[["text", "label_num"]]

        # Features and Labels
        X = df_data['text']
        y = df_data['label_num']

        # Split the data
        logger.info("Splitting dataset into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2,  # Using 80% for training, 20% for testing
            random_state=42, 
            stratify=y  # Ensure balanced split
        )

        # Create and fit the TF-IDF vectorizer
        logger.info("Creating and fitting TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(
            max_features=5000,  # Limit to top 5000 features
            min_df=2,          # Ignore terms that appear in less than 2 documents
            max_df=0.95,       # Ignore terms that appear in more than 95% of documents
            ngram_range=(1, 2) # Use both unigrams and bigrams
        )
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Initialize and train XGBoost
        logger.info("Training XGBoost model...")
        model = XGBClassifier(
            n_estimators=200,      # Number of boosting rounds
            max_depth=7,           # Maximum tree depth
            learning_rate=0.1,     # Learning rate
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(
            X_train_tfidf, 
            y_train,
            eval_set=[(X_test_tfidf, y_test)],
            early_stopping_rounds=20,
            verbose=True
        )

        # Evaluate the model
        logger.info("Evaluating model...")
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model Accuracy: {accuracy:.4f}")
        
        # Print detailed classification report
        report = classification_report(y_test, y_pred)
        logger.info(f"Classification Report:\n{report}")

        # Save the model and vectorizer
        logger.info("Saving model and vectorizer...")
        joblib.dump(model, 'spam_model_advanced.pkl')
        joblib.dump(vectorizer, 'vectorizer_advanced.pkl')
        
        logger.info("Model and vectorizer saved successfully!")
        return True

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        return False

if __name__ == "__main__":
    train_advanced_model() 