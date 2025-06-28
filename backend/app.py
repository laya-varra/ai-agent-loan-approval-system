from flask import Flask, request, jsonify
import os
from PIL import Image
import pytesseract
import uuid
import re
from datetime import datetime
from flask_cors import CORS
from abc import ABC, abstractmethod
from groq import Groq
import joblib  # For loading ML model
from dotenv import load_dotenv
import pandas as pd
# Load variables from .env file into environm
load_dotenv()


app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

# === Agent Classes ===

class Agent(ABC):
    def __init__(self, name):
        self.name = name
        self.status = "ready"
    @abstractmethod
    def process(self, data):
        pass
    def log_decision(self, message, decision_type="info"):
        return {
            "agent": self.name,
            "message": message,
            "type": decision_type,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

class DataValidationAgent(Agent):
    def __init__(self):
        super().__init__("Data Validation Agent")
        self.validation_rules = {
            'no_of_dependents': {'min': 0, 'max': 10, 'required': True},
            'income_annum': {'min': 1000, 'max': 10000000, 'required': True},
            'loan_amount': {'min': 1000, 'max': 5000000, 'required': True},
            'loan_term': {'min': 1, 'max': 480, 'required': True},
            'cibil_score': {'min': 300, 'max': 900, 'required': True},
            'residential_assets_value': {'min': 0, 'max': 50000000, 'required': False},
            'commercial_assets_value': {'min': 0, 'max': 50000000, 'required': False},
            'luxury_assets_value': {'min': 0, 'max': 10000000, 'required': False},
            'bank_asset_value': {'min': 0, 'max': 10000000, 'required': False}
        }
    def process(self, data):
        validation_results = []
        is_valid = True
        for field, rules in self.validation_rules.items():
            value = data.get(field)
            if rules['required'] and (value is None or value == ''):
                validation_results.append(self.log_decision(f"Missing required field: {field}", "error"))
                is_valid = False
                continue
            if value is not None and value != '':
                try:
                    numeric_value = float(value)
                    if numeric_value < rules['min'] or numeric_value > rules['max']:
                        validation_results.append(self.log_decision(
                            f"Field {field} value {numeric_value} is out of range ({rules['min']}-{rules['max']})", 
                            "error"
                        ))
                        is_valid = False
                except ValueError:
                    validation_results.append(self.log_decision(f"Invalid numeric value for {field}: {value}", "error"))
                    is_valid = False
        if is_valid:
            loan_amount = float(data.get('loan_amount', 0))
            income = float(data.get('income_annum', 0))
            if loan_amount > income * 10:
                validation_results.append(self.log_decision(
                    "Loan amount exceeds 10x annual income - flagged for review", 
                    "warning"
                ))
        if is_valid:
            validation_results.append(self.log_decision("All data validation checks passed", "success"))
        return {
            "valid": is_valid,
            "logs": validation_results,
            "processed_data": data if is_valid else None
        }

class DocumentVerificationAgent(Agent):
    def __init__(self):
        super().__init__("Document Verification Agent")
    def process(self, data):
        verification_results = []
        verification_score = 0
        income = float(data.get('income_annum', 0))
        cibil_score = int(data.get('cibil_score', 0))
        assets_total = (float(data.get('residential_assets_value', 0)) + 
                       float(data.get('commercial_assets_value', 0)) + 
                       float(data.get('luxury_assets_value', 0)) + 
                       float(data.get('bank_asset_value', 0)))
        if income > 0:
            if income < 50000:
                verification_results.append(self.log_decision("Income documents verified - Low income bracket", "info"))
                verification_score += 20
            elif income < 200000:
                verification_results.append(self.log_decision("Income documents verified - Medium income bracket", "success"))
                verification_score += 35
            else:
                verification_results.append(self.log_decision("Income documents verified - High income bracket", "success"))
                verification_score += 40
        if cibil_score >= 750:
            verification_results.append(self.log_decision("Excellent credit history verified", "success"))
            verification_score += 30
        elif cibil_score >= 650:
            verification_results.append(self.log_decision("Good credit history verified", "success"))
            verification_score += 20
        else:
            verification_results.append(self.log_decision("Poor credit history - requires additional verification", "warning"))
            verification_score += 5
        if assets_total > 0:
            verification_results.append(self.log_decision(f"Assets worth ${assets_total:,.2f} verified", "success"))
            verification_score += 20
        else:
            verification_results.append(self.log_decision("No assets declared", "info"))
        self_employed = data.get('self_employed', 'No')
        if self_employed == ' Yes':
            verification_results.append(self.log_decision("Self-employed status verified - requires additional income proof", "warning"))
            verification_score += 5
        else:
            verification_results.append(self.log_decision("Employment status verified", "success"))
            verification_score += 10
        verification_passed = verification_score >= 60
        if verification_passed:
            verification_results.append(self.log_decision(f"Document verification completed - Score: {verification_score}/100", "success"))
        else:
            verification_results.append(self.log_decision(f"Document verification incomplete - Score: {verification_score}/100", "error"))
        return {
            "verified": verification_passed,
            "verification_score": verification_score,
            "logs": verification_results
        }

class RuleBasedApprovalAgent(Agent):
    def __init__(self):
        super().__init__("Rule Based Approval Agent")
    def process(self, data):
        rule_results = []
        rule_score = 0
        rules_passed = 0
        total_rules = 6
        income = float(data.get('income_annum', 0))
        loan_amount = float(data.get('loan_amount', 0))
        cibil_score = int(data.get('cibil_score', 0))
        dependents = int(data.get('no_of_dependents', 0))
        loan_term = int(data.get('loan_term', 0))
        if income >= 25000:
            rule_results.append(self.log_decision("✓ Minimum income requirement met", "success"))
            rule_score += 20
            rules_passed += 1
        else:
            rule_results.append(self.log_decision("✗ Minimum income requirement not met (required: $25,000)", "error"))
        if cibil_score >= 600:
            rule_results.append(self.log_decision("✓ Minimum CIBIL score requirement met", "success"))
            rule_score += 25
            rules_passed += 1
        else:
            rule_results.append(self.log_decision("✗ Minimum CIBIL score requirement not met (required: 600)", "error"))
        dti_ratio = (loan_amount / income) * 100 if income > 0 else 100
        if dti_ratio <= 50:
            rule_results.append(self.log_decision(f"✓ Debt-to-Income ratio acceptable ({dti_ratio:.1f}%)", "success"))
            rule_score += 20
            rules_passed += 1
        else:
            rule_results.append(self.log_decision(f"✗ Debt-to-Income ratio too high ({dti_ratio:.1f}% > 50%)", "error"))
        max_loan = income * 8
        if loan_amount <= max_loan:
            rule_results.append(self.log_decision("✓ Loan amount within acceptable limits", "success"))
            rule_score += 15
            rules_passed += 1
        else:
            rule_results.append(self.log_decision(f"✗ Loan amount exceeds limit (max: ${max_loan:,.2f})", "error"))
        if dependents <= 4:
            rule_results.append(self.log_decision("✓ Number of dependents acceptable", "success"))
            rule_score += 10
            rules_passed += 1
        else:
            rule_results.append(self.log_decision("✗ Too many dependents for income level", "warning"))
        if 12 <= loan_term <= 360:
            rule_results.append(self.log_decision("✓ Loan term is reasonable", "success"))
            rule_score += 10
            rules_passed += 1
        else:
            rule_results.append(self.log_decision("✗ Loan term outside acceptable range (12-360 months)", "error"))
        rule_approval = rules_passed >= 4
        if rule_approval:
            rule_results.append(self.log_decision(f"Rule-based approval: APPROVED ({rules_passed}/{total_rules} rules passed)", "success"))
        else:
            rule_results.append(self.log_decision(f"Rule-based approval: REJECTED ({rules_passed}/{total_rules} rules passed)", "error"))
        return {
            "approved": rule_approval,
            "rules_passed": rules_passed,
            "total_rules": total_rules,
            "rule_score": rule_score,
            "logs": rule_results
        }

class GroqSuggestionAgent(Agent):
    def __init__(self):
        super().__init__("Groq Suggestion Agent")
        try:
            self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        except:
            self.client = None
    def process(self, data, prompt=None):
        if not self.client:
            return {
                "suggestion": "Groq API not available. Please set GROQ_API_KEY environment variable.",
                "logs": [self.log_decision("Groq API key not found", "error")]
            }
        try:
            if not prompt:
                prompt = "Suggest improvements for this loan application.: " + str(data)
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",  
            )
            suggestion = chat_completion.choices[0].message.content
            return {
                "suggestion": suggestion,
                "logs": [self.log_decision("Groq suggestion generated", "success")]
            }
        except Exception as e:
            return {
                "suggestion": f"Error generating suggestion: {str(e)}",
                "logs": [self.log_decision(f"Groq API error: {str(e)}", "error")]
            }

# === Machine Learning Agent ===

class MachineLearningAgent(Agent):
    def __init__(self, model_path="models/loan_decision_tree_20250518_053250.pkl"):
        super().__init__("Machine Learning Agent")
        try:
            self.model = joblib.load(model_path)
            self.status = "ready"
        except Exception as e:
            self.status = f"error: {str(e)}"
            self.model = None

    def process(self, data):
        if not self.model:
            return {
                "prediction": None,
                "logs": [self.log_decision("Model not loaded", "error")]
            }
        try:
            # Generate a synthetic loan_id
            loan_id = int(datetime.now().strftime('%Y%m%d%H%M%S') + str(uuid.uuid4().int)[:4])

            # Clean and encode education (assumes model expects 'Graduate' or 'Not Graduate')
            education_value = data.get('education', 'Graduate').strip().capitalize()

            # Construct input features as a DataFrame
            input_dict = {
                'loan_id': [loan_id],
                'no_of_dependents': [float(data['no_of_dependents'])],
                'education': [education_value],
                'income_annum': [float(data['income_annum'])],
                'loan_amount': [float(data['loan_amount'])],
                'loan_term': [float(data['loan_term'])],
                'cibil_score': [float(data['cibil_score'])],
                'residential_assets_value': [float(data['residential_assets_value'])],
                'commercial_assets_value': [float(data['commercial_assets_value'])],
                'luxury_assets_value': [float(data['luxury_assets_value'])],
                'bank_asset_value': [float(data['bank_asset_value'])],
                'self_employed': ['Yes' if data['self_employed'].strip() == 'Yes' else 'No']
            }

            df = pd.DataFrame(input_dict)

            # Predict using pipeline
            prediction = self.model.predict(df)[0]
            proba = self.model.predict_proba(df)[0][1]  # Probability of class "Approved"

            return {
                "prediction": "APPROVED" if prediction == " Approved" or prediction == 1 else "REJECTED",
                "confidence": float(proba),
                "logs": [
                    self.log_decision(
                        f"ML prediction: {'APPROVED' if prediction in [' Approved', 1] else 'REJECTED'} "
                        f"(Confidence: {proba:.2%})",
                        "success" if prediction in [' Approved', 1] else "warning"
                    )
                ]
            }

        except Exception as e:
            return {
                "prediction": None,
                "logs": [self.log_decision(f"Prediction error: {str(e)}", "error")]
            }




# === Initialize Agents ===

validation_agent = DataValidationAgent()
verification_agent = DocumentVerificationAgent()
approval_agent = RuleBasedApprovalAgent()
groq_agent = GroqSuggestionAgent()
ml_agent = MachineLearningAgent()

# === OCR Helper Functions ===

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_pan_format(pan_number):
    if not pan_number:
        return False
    pattern = r'^[A-Z]{5}[0-9]{4}[A-Z]$'
    return bool(re.match(pattern, pan_number))

def validate_aadhaar_format(aadhaar_number):
    if not aadhaar_number:
        return False
    cleaned = re.sub(r'\s+', '', aadhaar_number)
    return len(cleaned) == 12 and cleaned.isdigit()

def preprocess_image(image):
    if image.mode != 'L':
        image = image.convert('L')
    width, height = image.size
    if width < 800 or height < 600:
        scale_factor = max(800/width, 600/height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

def extract_text_with_ocr(image):
    configs = [
        r'--oem 3 --psm 6',
        r'--oem 3 --psm 7',
        r'--oem 3 --psm 8',
        r'--oem 3 --psm 11',
        r'--oem 3 --psm 12',
    ]
    best_text = ""
    best_confidence = 0
    for config in configs:
        try:
            text = pytesseract.image_to_string(image, config=config)
            confidence = len(text.strip()) * 0.1
            if confidence > best_confidence and len(text.strip()) > 10:
                best_text = text
                best_confidence = confidence
        except:
            continue
    return best_text if best_text else pytesseract.image_to_string(image)

# === Flask Routes ===



@app.route('/extract-pan', methods=['POST'])
def extract_pan():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if file and allowed_file(file.filename):
        try:
            image = Image.open(file.stream)
            image = preprocess_image(image)
            text = extract_text_with_ocr(image)
            pan_matches = re.findall(r'[A-Z]{5}[0-9]{4}[A-Z]', text)
            pan_number = pan_matches[0] if pan_matches else None
            valid = validate_pan_format(pan_number)
            return jsonify({
                "extracted_text": text,
                "pan_number": pan_number,
                "valid": valid
            })
        except Exception as e:
            return jsonify({"error": f"Failed to process image: {str(e)}"}), 500
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/extract-aadhaar', methods=['POST'])
def extract_aadhaar():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if file and allowed_file(file.filename):
        try:
            image = Image.open(file.stream)
            image = preprocess_image(image)
            text = extract_text_with_ocr(image)
            aadhaar_matches = re.findall(r'\d{4}\s?\d{4}\s?\d{4}', text)
            aadhaar_number = aadhaar_matches[0].replace(' ', '') if aadhaar_matches else None
            valid = validate_aadhaar_format(aadhaar_number)
            return jsonify({
                "extracted_text": text,
                "aadhaar_number": aadhaar_number,
                "valid": valid
            })
        except Exception as e:
            return jsonify({"error": f"Failed to process image: {str(e)}"}), 500
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/predict', methods=['POST'])
def process_loan():
    data = request.get_json()
    # Data Validation
    validation_result = validation_agent.process(data)
    if not validation_result['valid']:
        return jsonify({"error": "Validation failed", "logs": validation_result['logs']}), 400
    # Document Verification
    verification_result = verification_agent.process(validation_result['processed_data'])
    if not verification_result['verified']:
        return jsonify({"error": "Verification failed", "logs": verification_result['logs']}), 400
    # Rule-Based Approval
    approval_result = approval_agent.process(validation_result['processed_data'])
    # Machine Learning Prediction
    ml_result = ml_agent.process(validation_result['processed_data'])
    # Groq Suggestions
    '''groq_result = groq_agent.process(
        validation_result['processed_data'],
        f"Rule-based result: {approval_result['approved']}. ML prediction: {ml_result['prediction']}. Suggest improvements:"
    )'''
    groq_result = groq_agent.process(
    validation_result['processed_data'],
        f"""The rule-based result is: {approval_result['approved']}
        The ML prediction is: {ml_result['prediction']}

        Do the two decisions agree or not? Briefly explain in 1-2 sentences.
        and based on the two decisions whats your opinion
        DO NOT provide suggestions or recommendations."""
    )
    # Final decision logic
    final_decision = "REJECTED"
    if approval_result['approved'] and ml_result['prediction'] == "APPROVED":
        final_decision = "APPROVED"
    elif approval_result['approved'] or ml_result['prediction'] == "APPROVED":
        final_decision = "REQUIRES MANUAL REVIEW"
    return jsonify({
        "final_decision": final_decision,
        "rule_based": approval_result,
        "ml_prediction": ml_result,
        "groq_suggestion": groq_result,
        "verification_score": verification_result['verification_score']
    })

@app.route('/groq-suggest', methods=['POST'])
def groq_suggest():
    data = request.get_json()
    prompt = data.get('prompt', '')
    result = groq_agent.process({}, prompt)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)
