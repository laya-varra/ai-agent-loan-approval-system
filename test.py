import unittest
import joblib
import pandas as pd
from datetime import datetime
import uuid

class TestModelPrediction(unittest.TestCase):
    def setUp(self):
        self.model = joblib.load("models/loan_decision_tree_20250518_053250.pkl")

    def test_prediction_with_confidence(self):
        loan_id = int(datetime.now().strftime('%Y%m%d%H%M%S') + str(uuid.uuid4().int)[:4])
        input_data = {
            'loan_id': loan_id,
            'no_of_dependents': 1.0,
            'education': 'Graduate',
            'income_annum': 600000.0,
            'loan_amount': 200000.0,
            'loan_term': 120.0,
            'cibil_score': 720.0,
            'residential_assets_value': 500000.0,
            'commercial_assets_value': 300000.0,
            'luxury_assets_value': 0.0,
            'bank_asset_value': 150000.0,
            'self_employed': 'No'
        }

        df = pd.DataFrame([input_data])

        # Get prediction and probabilities
        prediction = self.model.predict(df)[0]
        proba = self.model.predict_proba(df)[0]  # array([P(rejected), P(approved)])

        cleaned_prediction = prediction.strip()
        confidence = proba[1] if cleaned_prediction == 'Approved' else proba[0]

        print("🧠 Raw prediction:", repr(prediction))
        print("✅ Cleaned:", cleaned_prediction)
        print("📊 Confidence score:", f"{confidence:.2%}")

        # Assertions
        self.assertIn(cleaned_prediction, ['Approved', 'Rejected'])
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

if __name__ == '__main__':
    unittest.main()
