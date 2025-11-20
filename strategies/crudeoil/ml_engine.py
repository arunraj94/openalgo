
import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from db_logger import init_db, Trade
import config

# Setup logger
logger = logging.getLogger("ML_ENGINE")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class MLEngine:
    def __init__(self, db_path=None, model_path="ml_model.pkl"):
        self.db_path = db_path or config.default_config.DB_PATH
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """Load trained model from disk if exists."""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                logger.info(f"Loaded ML model from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.model = None
        else:
            logger.warning("No ML model found. Please train first.")

    def fetch_data(self):
        """Fetch trade history from DB and prepare features."""
        session = init_db(self.db_path)
        try:
            trades = session.query(Trade).filter(Trade.realized_pnl != None).all()
            if not trades:
                logger.warning("No closed trades found in DB.")
                return pd.DataFrame()
            
            data = []
            for t in trades:
                data.append({
                    'entry_time': t.entry_time,
                    'pnl': t.realized_pnl
                })
            return pd.DataFrame(data)
        finally:
            session.close()

    def prepare_features(self, df):
        """Convert raw data into features."""
        if df.empty:
            return pd.DataFrame(), pd.Series()

        # Feature Engineering
        df['hour'] = df['entry_time'].dt.hour
        df['minute'] = df['entry_time'].dt.minute
        df['day_of_week'] = df['entry_time'].dt.dayofweek
        df['minute_of_day'] = df['hour'] * 60 + df['minute']
        
        # Target: 1 if PnL > 0, else 0
        df['target'] = (df['pnl'] > 0).astype(int)
        
        features = df[['day_of_week', 'minute_of_day']]
        target = df['target']
        
        return features, target

    def train(self):
        """Train the model on DB data."""
        logger.info("Starting model training...")
        raw_df = self.fetch_data()
        if raw_df.empty:
            logger.error("Training aborted: No data.")
            return False

        X, y = self.prepare_features(raw_df)
        if len(X) < 50:
            logger.warning(f"Not enough data to train (Found {len(X)} records). Need 50+.")
            return False

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model: Random Forest
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        
        logger.info(f"Training Complete. Accuracy: {acc:.2f}, Precision: {prec:.2f}")

        # Save
        joblib.dump(self.model, self.model_path)
        logger.info(f"Model saved to {self.model_path}")
        return True

    def predict(self, entry_time=None):
        """
        Predict probability of winning for a trade at `entry_time`.
        If entry_time is None, uses current time.
        Returns: Confidence Score (0.0 to 1.0)
        """
        if self.model is None:
            logger.warning("Model not loaded. Returning default confidence 1.0")
            return 1.0

        if entry_time is None:
            entry_time = datetime.now()

        # Create feature vector
        day_of_week = entry_time.weekday()
        minute_of_day = entry_time.hour * 60 + entry_time.minute
        
        features = pd.DataFrame([[day_of_week, minute_of_day]], columns=['day_of_week', 'minute_of_day'])
        
        try:
            # Get probability of class 1 (Win)
            probs = self.model.predict_proba(features)
            confidence = probs[0][1] # Probability of class 1
            return float(confidence)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 1.0

if __name__ == "__main__":
    # Test Run
    engine = MLEngine()
    engine.train()
    conf = engine.predict()
    print(f"Current Confidence: {conf:.2f}")
