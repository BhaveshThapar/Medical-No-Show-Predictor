# Medical Appointment No-Show Predictor

A comprehensive machine learning system that predicts patient no-show probability for medical appointments, featuring a complete ML pipeline, REST API, and interactive web dashboard.

## Project Overview

This project demonstrates end-to-end machine learning engineering skills by solving a real healthcare problem: predicting which patients are likely to miss their medical appointments. The system helps healthcare providers optimize scheduling and reduce wasted resources.

### Key Features
- **Complete ML Pipeline**: Data preprocessing, feature engineering, model training, and evaluation
- **Multiple ML Models**: Random Forest, Gradient Boosting, Logistic Regression, and SVM with automated model selection
- **Production-Ready API**: Flask REST API for real-time predictions
- **Interactive Dashboard**: Professional medical-themed web interface with real-time visualizations
- **Comprehensive Analytics**: ROC curves, feature importance analysis, and performance metrics

## Technical Stack

### Machine Learning & Data Science
- **Python 3.8+**
- **Scikit-learn**: ML models and preprocessing
- **Pandas & NumPy**: Data manipulation and analysis
- **Matplotlib & Plotly**: Data visualization

### Web Development & API
- **Flask**: Web framework and REST API
- **HTML5/CSS3/JavaScript**: Frontend development
- **Bootstrap 5**: Responsive UI framework
- **Plotly.js**: Interactive charts and graphs

### DevOps & Deployment
- **Joblib/Pickle**: Model serialization
- **RESTful API Design**: Industry-standard API endpoints
- **Responsive Design**: Mobile-friendly interface

## Installation & Setup

### Prerequisites
```bash
Python 3.8 or higher
pip package manager
```

### Quick Start
1. **Clone the repository**
```bash
git clone https://github.com/bhaveshthapar/medical-noshow-predictor
cd medical-noshow-predictor
```

2. **Install dependencies**
```bash
pip install pandas numpy scikit-learn flask plotly joblib
```

3. **Run the application**
```bash
python noshow_predictor.py
```

4. **Access the dashboard**
```
Open your browser to: http://localhost:5000
```

## Dataset & Features

### Original Data Structure
The system works with medical appointment datasets containing:
- **Patient Demographics**: Age, Gender, Location
- **Medical Conditions**: Hypertension, Diabetes, Alcoholism, Handicap status
- **Appointment Details**: Scheduling date, appointment date, SMS notifications
- **Socioeconomic Factors**: Scholarship status

### Engineered Features
The system automatically creates additional predictive features:
- **Time-based Features**: Days between scheduling and appointment, weekday patterns
- **Risk Scores**: Composite health and behavioral risk indicators
- **Age Groups**: Categorical age segmentation
- **Condition Aggregation**: Total number of health conditions

## Machine Learning Pipeline

### 1. Data Preprocessing
- **Automated Data Cleaning**: Handles missing values and outliers
- **Feature Engineering**: Creates 8+ additional predictive features
- **Encoding**: Categorical variable encoding with label encoders
- **Scaling**: StandardScaler for numerical features

### 2. Model Training & Selection
- **Multiple Algorithms**: Tests 4 different ML algorithms
- **Cross-Validation**: Ensures robust model evaluation
- **Automated Selection**: Chooses best model based on ROC-AUC score
- **Hyperparameter Optimization**: GridSearchCV for optimal parameters

### 3. Model Evaluation
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualization**: ROC curves and feature importance plots
- **Performance Monitoring**: Real-time metrics dashboard

## API Documentation

### Endpoints

#### `GET /`
- **Description**: Main dashboard interface
- **Returns**: HTML dashboard with model performance and prediction form

#### `POST /predict`
- **Description**: Single patient no-show prediction
- **Content-Type**: `application/json`
- **Request Body**:
```json
{
  "Age": 35,
  "Gender": "F",
  "SMS_received": 1,
  "Scholarship": 0,
  "Hipertension": 0,
  "Diabetes": 0,
  "Alcoholism": 0,
  "Handcap": 0,
  "days_between": 5
}
```
- **Response**:
```json
{
  "no_show_probability": 0.234,
  "prediction": 0,
  "risk_level": "Low"
}
```

#### `GET /model_performance`
- **Description**: Retrieve model performance metrics
- **Returns**: JSON with accuracy, precision, recall, ROC-AUC, and visualization data

#### `POST /train`
- **Description**: Trigger model retraining
- **Returns**: Training results and best model information

## Dashboard Features

### Performance Metrics
- **Real-time KPIs**: Accuracy, Precision, Recall, ROC-AUC scores
- **Interactive Charts**: ROC curve analysis and feature importance visualization
- **Model Comparison**: Performance across different algorithms

### Patient Risk Assessment
- **Interactive Form**: Easy-to-use patient data input
- **Real-time Prediction**: Instant no-show probability calculation
- **Risk Categorization**: Color-coded risk levels (Low/Medium/High)
- **Medical Theming**: Professional healthcare-focused design

## Technical Challenges Solved

1. **Class Imbalance**: Implemented stratified sampling and appropriate metrics for imbalanced healthcare data
2. **Feature Engineering**: Created domain-specific features from temporal and categorical data
3. **Model Selection**: Automated comparison of multiple algorithms with cross-validation
4. **Production Deployment**: Built scalable API with error handling and monitoring
5. **User Experience**: Created intuitive medical-themed interface for healthcare professionals

## Future Enhancements

- **Database Integration**: PostgreSQL/MongoDB for data persistence
- **Advanced ML**: Deep learning models and ensemble methods
- **Real-time Processing**: Kafka/Redis for streaming predictions
- **Cloud Deployment**: AWS/GCP deployment with Docker containers
- **A/B Testing**: Framework for model performance comparison
- **Security**: Authentication and data encryption for HIPAA compliance

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*This project demonstrates production-ready machine learning engineering skills applicable to healthcare technology, fintech, and data-driven industries.*