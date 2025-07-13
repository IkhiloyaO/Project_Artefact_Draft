import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import joblib

# Load the Data
filepath = r'C:\Users\HP win10\Desktop\balanced_sample_water_potability.csv'
wpdata = pd.read_csv(filepath)

# Exploratory Data Analysis
print("\n Basic Info:")
print(wpdata.info())
print("\n Descriptive Stats:")
print(wpdata.describe())
print("\n Missing Values:")
print(wpdata.isnull().sum())

plt.figure(figsize=(12, 6))
sns.heatmap(wpdata.drop('Potability_Status', axis=1).corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Prepare Data
X = wpdata.drop('Potability_Status', axis=1)
y = wpdata['Potability_Status']

# Combine Rare Classes
threshold = 2
value_counts = y.value_counts()
rare_classes = value_counts[value_counts < threshold].index
y = y.replace(rare_classes, "Other Health Risks Including GI Illnesses, Cancer and Hepatitis")

print("Class Distribution After Combining Rare Classes:")
print(y.value_counts())

# Check to confirm:
print("\nFiltered Class Distribution:")
print(y.value_counts())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')    # Save the scaler

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Model Selection
models = {
    "SVM": SVC(class_weight='balanced'),
    "Random Forest": RandomForestClassifier(random_state=42),
    "ANN": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
}

results = []
print("\n Model Evaluation (w/ Cross Validation):\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted')

    print(f"\n{name} Report:")
    print(classification_report(y_test, y_pred))

    results.append((name, acc, f1, rec, prec,))

# Result Summary
    print("\n Model Comparison:")
    summary = pd.DataFrame(results, columns=["Model", "Accuracy", "F1 Score", "Recall", "Precision"])
    print(summary.sort_values(by="F1 Score", ascending=False))

# Define hyperparameter grid
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Use Random Forest
rf = RandomForestClassifier(random_state=42)
cv_strategy = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,
    scoring='f1_weighted',
    cv=cv_strategy,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("\nBest Random Forest Parameters:")
print(random_search.best_params_)

# Evaluate Best RF on test data
best_rf = random_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)

print("\nClassification Report (Tuned Random Forest):")
print(classification_report(y_test, y_pred_rf))

# Save best tuned model
joblib.dump(best_rf, "rf_model.pkl")
print("\n Best tuned model 'Random Forest' saved.")
