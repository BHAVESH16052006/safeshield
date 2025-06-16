import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
from transformers import BertTokenizer, BertModel
from xgboost import XGBClassifier
import joblib
import logging
from tqdm import tqdm
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def get_bert_embeddings(texts, tokenizer, model, max_length=128):
    """Extract BERT embeddings for given texts."""
    embeddings = []
    
    for text in tqdm(texts, desc="Extracting BERT embeddings"):
        # Tokenize and encode text
        inputs = tokenizer(text, 
                         max_length=max_length,
                         padding='max_length',
                         truncation=True,
                         return_tensors='pt')
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(embedding[0])
    
    return np.array(embeddings)

def train_hybrid_model():
    try:
        # Load BERT tokenizer and model
        logger.info("Loading BERT model and tokenizer...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        bert_model = bert_model.to(device)
        bert_model.eval()

        # Load the dataset
        logger.info("Loading dataset...")
        df = pd.read_csv("spam_ham_dataset.csv")
        df_data = df[["text", "label_num"]]

        # Split the data
        logger.info("Splitting dataset...")
        X_train, X_test, y_train, y_test = train_test_split(
            df_data['text'], 
            df_data['label_num'],
            test_size=0.2,
            random_state=42,
            stratify=df_data['label_num']
        )

        # Get BERT embeddings for training and test sets
        logger.info("Generating BERT embeddings for training set...")
        X_train_bert = get_bert_embeddings(X_train, tokenizer, bert_model)
        logger.info("Generating BERT embeddings for test set...")
        X_test_bert = get_bert_embeddings(X_test, tokenizer, bert_model)

        # Train XGBoost on BERT embeddings
        logger.info("Training XGBoost model...")
        xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        xgb_model.fit(
            X_train_bert, 
            y_train,
            eval_set=[(X_test_bert, y_test)],
            early_stopping_rounds=20,
            verbose=True
        )

        # Make predictions
        logger.info("Evaluating model...")
        y_pred = xgb_model.predict(X_test_bert)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model Accuracy: {accuracy:.4f}")
        
        # Print detailed classification report
        report = classification_report(y_test, y_pred)
        logger.info(f"Classification Report:\n{report}")

        # Save the models
        logger.info("Saving models...")
        # Save XGBoost model
        joblib.dump(xgb_model, 'spam_model_hybrid.pkl')
        # Save BERT tokenizer and model
        tokenizer.save_pretrained('./bert_tokenizer')
        bert_model.save_pretrained('./bert_model')
        
        logger.info("Models saved successfully!")
        return True

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        return False

if __name__ == "__main__":
    train_hybrid_model() 