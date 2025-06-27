import unittest
from datetime import datetime
from app import MachineLearningAgent  # Update if needed based on your filename

class TestMachineLearningAgentWithModel(unittest.TestCase):
    def setUp(self):
        self.agent = MachineLearningAgent(
            model_path="models/loan_decision_tree_20250518_053250.pkl"
        )
        self.assertIsNotNone(self.agent.model, "Model failed to load!")

        self.valid_data = {
            'no_of_dependents': '1',
            'education': 'Graduate',
            'income_annum': '600000',
            'loan_amount': '200000',
            'loan_term': '120',
            'cibil_score': '720',
            'residential_assets_value': '500000',
            'commercial_assets_value': '300000',
            'luxury_assets_value': '0',
            'bank_asset_value': '150000',
            'self_employed': ' No'  # Note the space; match actual form input if needed
        }

    def test_real_prediction(self):
        result = self.agent.process(self.valid_data)

        print("\n🔍 Full Prediction Output:")
        print(result)

        # Validate prediction output
        self.assertIn(result['prediction'], ['APPROVED', 'REJECTED'], "Prediction should be APPROVED or REJECTED")

        # Validate confidence value
        self.assertIsInstance(result['confidence'], float, "Confidence must be a float")
        self.assertGreaterEqual(result['confidence'], 0.0, "Confidence must be >= 0.0")
        self.assertLessEqual(result['confidence'], 1.0, "Confidence must be <= 1.0")

        # Validate logs
        self.assertIsInstance(result['logs'], list)
        self.assertGreater(len(result['logs']), 0, "Logs should contain at least one entry")

        # Optionally check timestamp format
        log_entry = result['logs'][0]
        self.assertIn('timestamp', log_entry)
        datetime.strptime(log_entry['timestamp'], "%Y-%m-%d %H:%M:%S")  # Check format

if __name__ == '__main__':
    unittest.main()
