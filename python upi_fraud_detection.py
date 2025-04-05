import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load Dataset (Replace with actual dataset)
data = pd.read_csv("upi_transactions.csv")

# Data Preprocessing
data.fillna(method='ffill', inplace=True)

# Feature Engineering
data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
data['is_night'] = data['hour'].apply(lambda x: 1 if x < 6 or x > 22 else 0)

# Selecting Features & Target
features = ['amount', 'is_night', 'transaction_frequency', 'device_mismatch']
target = 'fraud_label'
X = data[features]
y = data[target]

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_prob))

# Save Model
import joblib
joblib.dump(model, "upi_fraud_model.pkl")

# Flask API for Deployment
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({'fraud_prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
