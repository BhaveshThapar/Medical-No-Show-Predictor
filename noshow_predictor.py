import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           accuracy_score, precision_score, recall_score, f1_score)

from flask import Flask, request, jsonify, render_template_string
import plotly.graph_objs as go
import plotly.utils
import json

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_and_clean_data(self, filepath):
        
        np.random.seed(42)
        n_samples = 10000
        
        ages = np.random.exponential(scale=25, size=n_samples)
        ages = np.clip(ages, 0, 95).astype(int)
        
        genders = np.random.choice(['M', 'F'], n_samples)
        
        age_factor = ages / 100.0
        hipertension = np.random.binomial(1, np.clip(age_factor * 0.4 + 0.1, 0, 0.8), n_samples)
        diabetes = np.random.binomial(1, np.clip(age_factor * 0.3 + 0.05, 0, 0.4), n_samples)
        alcoholism = np.random.binomial(1, 0.05, n_samples)

        scholarship_prob = np.clip(0.3 - age_factor * 0.2, 0.05, 0.3)
        scholarship = np.random.binomial(1, scholarship_prob, n_samples)
        
        handcap = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.92, 0.05, 0.02, 0.008, 0.002])
        
        sms_received = np.random.binomial(1, 0.68, n_samples)
        
        days_between = np.random.exponential(scale=7, size=n_samples)
        days_between = np.clip(days_between, 0, 179).astype(int)
        
        scheduled_weekday = np.random.choice(range(7), n_samples)
        appointment_weekday = np.random.choice(range(7), n_samples)
        
        no_show_base_prob = 0.2
        
        age_effect = np.where(ages < 18, 0.15, np.where(ages > 65, -0.05, 0))  
        sms_effect = np.where(sms_received == 0, 0.25, -0.05) 
        days_effect = np.clip(days_between * 0.01, 0, 0.3) 
        health_effect = -(hipertension + diabetes) * 0.05 
        weekend_effect = np.where((appointment_weekday == 5) | (appointment_weekday == 6), 0.1, 0) 
        scholarship_effect = np.where(scholarship == 1, 0.08, 0)
        
        no_show_prob = np.clip(
            no_show_base_prob + age_effect + sms_effect + days_effect + 
            health_effect + weekend_effect + scholarship_effect,
            0.05, 0.85
        )
        
        no_show = np.random.binomial(1, no_show_prob, n_samples)
        
        df = pd.DataFrame({
            'PatientId': np.random.randint(1000000, 9999999, n_samples),
            'AppointmentID': np.random.randint(5000000, 6000000, n_samples),
            'Gender': genders,
            'Age': ages,
            'Neighbourhood': np.random.choice(['JARDIM DA PENHA', 'MATA DA PRAIA', 'PONTAL DE CAMBURI'], n_samples),
            'Scholarship': scholarship,
            'Hipertension': hipertension,
            'Diabetes': diabetes,
            'Alcoholism': alcoholism,
            'Handcap': handcap,
            'SMS_received': sms_received,
            'days_between': days_between,
            'scheduled_weekday': scheduled_weekday,
            'appointment_weekday': appointment_weekday,
            'No-show': no_show
        })
        
        base_date = datetime(2016, 4, 29)
        df['ScheduledDay'] = [base_date + timedelta(days=np.random.randint(-30, 1)) for _ in range(n_samples)]
        df['AppointmentDay'] = [scheduled + timedelta(days=int(days)) for scheduled, days in zip(df['ScheduledDay'], df['days_between'])]
        
        print(f"Generated {n_samples} synthetic appointments")
        print(f"No-show rate: {df['No-show'].mean():.1%}")
        print(f"Age range: {df['Age'].min()}-{df['Age'].max()}")
        print(f"Days between range: {df['days_between'].min()}-{df['days_between'].max()}")
        
        return df
    
    def feature_engineering(self, df):
        df = df.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(df['ScheduledDay']):
            df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
        if not pd.api.types.is_datetime64_any_dtype(df['AppointmentDay']):
            df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
        
        if 'scheduled_hour' not in df.columns:
            df['scheduled_hour'] = df['ScheduledDay'].dt.hour
        
        df['age_group'] = pd.cut(df['Age'], bins=[0, 18, 35, 50, 65, 100], 
                                labels=['child', 'young_adult', 'adult', 'middle_aged', 'senior'])
        
        df['total_conditions'] = (df['Hipertension'] + df['Diabetes'] + 
                                 df['Alcoholism'] + (df['Handcap'] > 0).astype(int))
        
        df['risk_score'] = (
            (df['Age'] < 18).astype(int) * 0.3 + 
            (df['Age'] > 80).astype(int) * 0.1 + 
            (1 - df['SMS_received']) * 0.4 +    
            np.clip(df['days_between'] / 30.0, 0, 1) * 0.3 + 
            df['Scholarship'] * 0.2               
        )
        
        df['is_weekend'] = ((df['appointment_weekday'] == 5) | (df['appointment_weekday'] == 6)).astype(int)
        
        return df
    
    def prepare_features(self, df, fit_encoders=True):
        feature_cols = ['Age', 'Gender', 'Scholarship', 'Hipertension', 'Diabetes', 
                       'Alcoholism', 'Handcap', 'SMS_received', 'days_between',
                       'scheduled_weekday', 'appointment_weekday', 'total_conditions', 
                       'risk_score', 'is_weekend']
        
        X = df[feature_cols].copy()
        
        if fit_encoders:
            self.label_encoders['Gender'] = LabelEncoder()
            X['Gender'] = self.label_encoders['Gender'].fit_transform(X['Gender'])
        else:
            if 'Gender' in self.label_encoders:
                X['Gender'] = self.label_encoders['Gender'].transform(X['Gender'])
            else:
                X['Gender'] = X['Gender'].map({'F': 0, 'M': 1}).fillna(0)
        
        y = df['No-show'] if 'No-show' in df.columns else None
        
        return X, y

class NoShowPredictor:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.preprocessor = DataPreprocessor()
        
    def train_models(self, X_train, y_train, X_test, y_test):
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced',
                C=1.0
            ),
            'SVM': SVC(
                probability=True, 
                random_state=42,
                class_weight='balanced',
                C=1.0,
                kernel='rbf'
            )
        }
        
        results = {}
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        print(f"Training no-show rate: {y_train.mean():.1%}")
        print(f"Test no-show rate: {y_test.mean():.1%}")
        print()
        
        for name, model in models.items():
            print(f"Testing {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            results[name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            print(f"  {name}: AUC = {metrics['roc_auc']:.4f}, F1 = {metrics['f1']:.4f}, Precision = {metrics['precision']:.4f}, Recall = {metrics['recall']:.4f}")
            print(f"    Predicted {y_pred.sum()} no-shows out of {len(y_pred)} appointments")
        
        best_model_name = max(results.keys(), key=lambda x: results[x]['metrics']['roc_auc'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        self.models = results
        
        print(f"\n🏆 Winner: {best_model_name} performs best!")
        return results
    
    def get_feature_importance(self, feature_names):
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            return pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        elif hasattr(self.best_model, 'coef_'):
            importance = np.abs(self.best_model.coef_[0])
            return pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        return None
    
    def predict_single(self, patient_data):
        if self.best_model is None:
            raise ValueError("Need to train the model first!")
        X, _ = self.preprocessor.prepare_features(pd.DataFrame([patient_data]), fit_encoders=False)
        probability = self.best_model.predict_proba(X)[0, 1]
        prediction = self.best_model.predict(X)[0]
        return {
            'no_show_probability': float(probability),
            'prediction': int(prediction),
            'risk_level': 'High' if probability > 0.6 else 'Medium' if probability > 0.3 else 'Low'
        }

app = Flask(__name__)
predictor = NoShowPredictor()
model_results = {}
X_test_global = None
y_test_global = None

print("Medical Appointment No-Show Predictor")
print("=" * 50)
print("Training model with synthetic data...")
df = predictor.preprocessor.load_and_clean_data("synthetic_data.csv")
df = predictor.preprocessor.feature_engineering(df)
X, y = predictor.preprocessor.prepare_features(df, fit_encoders=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_test_global = X_test
y_test_global = y_test
model_results = predictor.train_models(X_train, y_train, X_test, y_test)

@app.route('/')
def dashboard():
    dashboard_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Medical Appointment No-Show Predictor Dashboard</title>
        <meta name='viewport' content='width=device-width, initial-scale=1'>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.18.0/plotly.min.js"></script>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
            body { background: linear-gradient(135deg, #e3f2fd 0%, #f8f9ff 100%); font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            .container-fluid { padding-left: 0.5rem; padding-right: 0.5rem; }
            .metric-card { 
                background: linear-gradient(135deg, #1565c0 0%, #0d47a1 100%); 
                color: white; 
                border-radius: 15px;
                box-shadow: 0 8px 20px rgba(21, 101, 192, 0.3);
                border-left: 5px solid #00c853;
                transition: transform 0.2s ease;
                margin-bottom: 1rem;
            }
            .metric-card:hover { transform: translateY(-5px); }
            .prediction-card { 
                background: linear-gradient(135deg, #d32f2f 0%, #b71c1c 100%); 
                color: white;
                border-radius: 15px;
                box-shadow: 0 8px 20px rgba(211, 47, 47, 0.3);
                border-left: 5px solid #ff5722;
                margin-bottom: 1rem;
            }
            .chart-container { 
                background: white; 
                border-radius: 15px; 
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                border-left: 5px solid #4caf50;
                transition: box-shadow 0.3s ease;
                margin-bottom: 1rem;
            }
            .chart-container:hover { box-shadow: 0 12px 35px rgba(0,0,0,0.15); }
            h1 { 
                color: #1565c0; 
                font-weight: 700;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 2rem;
                font-size: 2rem;
            }
            @media (max-width: 767px) {
                h1 { font-size: 1.3rem; }
                .metric-card, .prediction-card, .chart-container, .card { border-radius: 10px; }
                .card-header { border-radius: 10px 10px 0 0 !important; }
                .container-fluid { padding-left: 0.2rem; padding-right: 0.2rem; }
            }
            .card { 
                border: none;
                border-radius: 15px;
                box-shadow: 0 6px 20px rgba(0,0,0,0.08);
                transition: all 0.3s ease;
                margin-bottom: 1rem;
            }
            .card:hover { box-shadow: 0 10px 30px rgba(0,0,0,0.12); }
            .card-header { 
                background: linear-gradient(135deg, #4caf50 0%, #388e3c 100%);
                color: white;
                border-radius: 15px 15px 0 0 !important;
                border: none;
                font-weight: 600;
            }
            .form-control, .form-select {
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 10px 15px;
                transition: all 0.3s ease;
            }
            .form-control:focus, .form-select:focus {
                border-color: #4caf50;
                box-shadow: 0 0 0 0.2rem rgba(76, 175, 80, 0.25);
            }
            .btn-primary {
                background: linear-gradient(135deg, #4caf50 0%, #388e3c 100%);
                border: none;
                border-radius: 25px;
                padding: 12px 30px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1px;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
            }
            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
                background: linear-gradient(135deg, #388e3c 0%, #2e7d32 100%);
            }
            .alert {
                border-radius: 10px;
                border: none;
                font-weight: 600;
            }
            .medical-icon {
                font-size: 1.2em;
                margin-right: 8px;
                color: #4caf50;
            }
            label {
                font-weight: 600;
                color: #1565c0;
                margin-bottom: 8px;
                display: block;
            }
            .metric-value {
                font-size: 2.5rem;
                font-weight: 700;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
            }
            .metric-label {
                font-size: 0.9rem;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 1px;
                opacity: 0.9;
            }
        </style>
    </head>
    <body class="bg-light">
        <div class="container-fluid py-4">
            <h1 class="text-center mb-4">Medical Appointment No-Show Predictor</h1>
            <div class="row mb-4">
                <div class="col-6 col-md-3">
                    <div class="card metric-card text-center p-3">
                        <h3 id="accuracy">-</h3>
                        <p>Accuracy</p>
                    </div>
                </div>
                <div class="col-6 col-md-3">
                    <div class="card metric-card text-center p-3">
                        <h3 id="precision">-</h3>
                        <p>Precision</p>
                    </div>
                </div>
                <div class="col-6 col-md-3">
                    <div class="card metric-card text-center p-3">
                        <h3 id="recall">-</h3>
                        <p>Recall</p>
                    </div>
                </div>
                <div class="col-6 col-md-3">
                    <div class="card metric-card text-center p-3">
                        <h3 id="roc_auc">-</h3>
                        <p>ROC AUC</p>
                    </div>
                </div>
            </div>
            <div class="row mb-4">
                <div class="col-12 col-md-6">
                    <div class="card chart-container p-3">
                        <h5>ROC Curve</h5>
                        <div id="roc-curve"></div>
                    </div>
                </div>
                <div class="col-12 col-md-6">
                    <div class="card chart-container p-3">
                        <h5>Feature Importance</h5>
                        <div id="feature-importance"></div>
                    </div>
                </div>
            </div>
            <div class="row">
                <div class="col-12 col-lg-8">
                    <div class="card">
                        <div class="card-header">
                            <h5>Single Patient Prediction</h5>
                        </div>
                        <div class="card-body">
                            <form id="prediction-form">
                                <div class="row">
                                    <div class="col-12 col-md-4">
                                        <div class="mb-3">
                                            <label>Age</label>
                                            <input type="number" class="form-control" name="Age" value="35" min="0" max="100">
                                        </div>
                                        <div class="mb-3">
                                            <label>Gender</label>
                                            <select class="form-control" name="Gender">
                                                <option value="F">Female</option>
                                                <option value="M">Male</option>
                                            </select>
                                        </div>
                                        <div class="mb-3">
                                            <label>Days Between Scheduling & Appointment</label>
                                            <input type="number" class="form-control" name="days_between" value="5" min="0">
                                        </div>
                                    </div>
                                    <div class="col-12 col-md-4">
                                        <div class="mb-3">
                                            <label>SMS Received</label>
                                            <select class="form-control" name="SMS_received">
                                                <option value="1">Yes</option>
                                                <option value="0">No</option>
                                            </select>
                                        </div>
                                        <div class="mb-3">
                                            <label>Scholarship</label>
                                            <select class="form-control" name="Scholarship">
                                                <option value="0">No</option>
                                                <option value="1">Yes</option>
                                            </select>
                                        </div>
                                        <div class="mb-3">
                                            <label>Hypertension</label>
                                            <select class="form-control" name="Hipertension">
                                                <option value="0">No</option>
                                                <option value="1">Yes</option>
                                            </select>
                                        </div>
                                    </div>
                                    <div class="col-12 col-md-4">
                                        <div class="mb-3">
                                            <label>Diabetes</label>
                                            <select class="form-control" name="Diabetes">
                                                <option value="0">No</option>
                                                <option value="1">Yes</option>
                                            </select>
                                        </div>
                                        <div class="mb-3">
                                            <label>Alcoholism</label>
                                            <select class="form-control" name="Alcoholism">
                                                <option value="0">No</option>
                                                <option value="1">Yes</option>
                                            </select>
                                        </div>
                                        <div class="mb-3">
                                            <label>Handicap Level</label>
                                            <select class="form-control" name="Handcap">
                                                <option value="0">0</option>
                                                <option value="1">1</option>
                                                <option value="2">2</option>
                                                <option value="3">3</option>
                                                <option value="4">4</option>
                                            </select>
                                        </div>
                                    </div>
                                </div>
                                <button type="submit" class="btn btn-primary w-100 mt-2">Predict No-Show Risk</button>
                            </form>
                        </div>
                    </div>
                </div>
                <div class="col-12 col-lg-4">
                    <div class="card prediction-card">
                        <div class="card-body text-center">
                            <h5>Prediction Result</h5>
                            <div id="prediction-result">
                                <p>Enter patient data and click predict</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Load model performance data
            fetch('/model_performance')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        console.error('Model not trained:', data.error);
                        return;
                    }
                    
                    document.getElementById('accuracy').textContent = (data.accuracy * 100).toFixed(1) + '%';
                    document.getElementById('precision').textContent = (data.precision * 100).toFixed(1) + '%';
                    document.getElementById('recall').textContent = (data.recall * 100).toFixed(1) + '%';
                    document.getElementById('roc_auc').textContent = data.roc_auc.toFixed(3);
                    
                    // Plot ROC curve
                    if (data.roc_curve_data && data.roc_curve_data.length > 0) {
                        Plotly.newPlot('roc-curve', data.roc_curve_data, {
                            title: 'ROC Curve',
                            xaxis: { title: 'False Positive Rate' },
                            yaxis: { title: 'True Positive Rate' }
                        });
                    }
                    
                    // Plot feature importance
                    if (data.feature_importance_data && data.feature_importance_data.length > 0) {
                        Plotly.newPlot('feature-importance', data.feature_importance_data, {
                            title: 'Feature Importance',
                            xaxis: { title: 'Importance' },
                            yaxis: { title: 'Features' },
                            margin: { l: 120 }
                        });
                    } else {
                        document.getElementById('feature-importance').innerHTML = '<p class="text-muted">Feature importance not available for this model type</p>';
                    }
                })
                .catch(error => {
                    console.error('Error loading model performance:', error);
                });

            // Handle prediction form
            document.getElementById('prediction-form').addEventListener('submit', function(e) {
                e.preventDefault();
                
                const formData = new FormData(e.target);
                const data = Object.fromEntries(formData.entries());
                
                // Convert numeric fields
                ['Age', 'days_between', 'SMS_received', 'Scholarship', 'Hipertension', 
                 'Diabetes', 'Alcoholism', 'Handcap'].forEach(field => {
                    data[field] = parseInt(data[field]);
                });
                
                // Add calculated fields
                data['scheduled_weekday'] = 1; // Monday
                data['appointment_weekday'] = 1; // Monday  
                data['total_conditions'] = data['Hipertension'] + data['Diabetes'] + data['Alcoholism'] + (data['Handcap'] > 0 ? 1 : 0);
                data['risk_score'] = (data['Age'] < 18 ? 0.3 : 0) + (data['Age'] > 80 ? 0.1 : 0) + 
                                   (1 - data['SMS_received']) * 0.4 + Math.min(data['days_between'] / 30.0, 1) * 0.3 + 
                                   data['Scholarship'] * 0.2;
                data['is_weekend'] = 0; // Monday is not weekend
                
                fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                })
                .then(response => response.json())
                .then(result => {
                    if (result.error) {
                        document.getElementById('prediction-result').innerHTML = `<div class="alert alert-danger">Error: ${result.error}</div>`;
                        return;
                    }
                    
                    document.getElementById('prediction-result').innerHTML = `
                        <h3>${(result.no_show_probability * 100).toFixed(1)}%</h3>
                        <p>No-Show Probability</p>
                        <div class="alert alert-${result.risk_level === 'High' ? 'danger' : 
                                                    result.risk_level === 'Medium' ? 'warning' : 'success'}">${result.risk_level} Risk</div>
                        <p class="mb-0">Prediction: ${result.prediction === 1 ? 'Will Not Show' : 'Will Show'}</p>
                    `;
                })
                .catch(error => {
                    console.error('Error making prediction:', error);
                    document.getElementById('prediction-result').innerHTML = `<div class="alert alert-danger">Error making prediction</div>`;
                });
            });
        </script>
    </body>
    </html>
    """
    return dashboard_html

@app.route('/model_performance')
def model_performance():
    global model_results, X_test_global, y_test_global
    
    if not model_results or predictor.best_model is None:
        return jsonify({'error': 'Model not trained yet'})
    
    best_result = model_results[predictor.best_model_name]
    metrics = best_result['metrics']
    
    roc_curve_data = []
    if y_test_global is not None:
        fpr, tpr, _ = roc_curve(y_test_global, best_result['probabilities'])
        roc_curve_data = [{
            'x': fpr.tolist(),
            'y': tpr.tolist(),
            'type': 'scatter',
            'mode': 'lines',
            'name': f'{predictor.best_model_name} (AUC = {metrics["roc_auc"]:.3f})',
            'line': {'color': 'blue', 'width': 3}
        }, {
            'x': [0, 1],
            'y': [0, 1],
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Random',
            'line': {'color': 'red', 'dash': 'dash'}
        }]
    
    feature_importance_data = []
    if X_test_global is not None:
        importance_df = predictor.get_feature_importance(X_test_global.columns)
        if importance_df is not None:
            top_features = importance_df.head(10)
            feature_importance_data = [{
                'x': top_features['importance'].tolist(),
                'y': top_features['feature'].tolist(),
                'type': 'bar',
                'orientation': 'h',
                'marker': {'color': 'green'}
            }]
    
    return jsonify({
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'roc_auc': metrics['roc_auc'],
        'best_model': predictor.best_model_name,
        'roc_curve_data': roc_curve_data,
        'feature_importance_data': feature_importance_data
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if predictor.best_model is None:
            return jsonify({'error': 'Model not trained yet'})
        
        patient_data = request.json
        result = predictor.predict_single(patient_data)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)