
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('Rainfall.csv')

# Display basic info
print(df.head())
print(df.shape)
print(df.info())
print(df.describe().T)
print(df.isnull().sum())

# Clean column names
df.rename(str.strip, axis='columns', inplace=True)

# Fill missing values with column mean
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

# Check total missing values after filling
print("Total missing values:", df.isnull().sum().sum())

# Pie chart of rainfall distribution
plt.pie(df['rainfall'].value_counts().values,
        labels=df['rainfall'].value_counts().index,
        autopct='%1.1f%%')
plt.title("Rainfall Distribution")
plt.show()

# Feature selection
features = list(df.select_dtypes(include=np.number).columns)
features.remove('day')
print("Selected numerical features:", features)

# Histograms
plt.subplots(figsize=(15, 8))
for i, col in enumerate(features):
    plt.subplot(3, 4, i + 1)
    sb.histplot(df[col], kde=True)
plt.tight_layout()
plt.show()

# Boxplots
plt.subplots(figsize=(15, 8))
for i, col in enumerate(features):
    plt.subplot(3, 4, i + 1)
    sb.boxplot(x=df[col])
plt.tight_layout()
plt.show()

# Encode target variable
df.replace({'yes': 1, 'no': 0}, inplace=True)

# Correlation heatmap
plt.figure(figsize=(10, 10))
sb.heatmap(df.corr() > 0.8, annot=True, cbar=False)
plt.title("High Correlation Matrix")
plt.show()

# Drop less relevant features
df.drop(['maxtemp', 'mintemp'], axis=1, inplace=True)

# Split into features and target
features = df.drop(['day', 'rainfall'], axis=1)
target = df.rainfall

X_train, X_val, Y_train, Y_val = train_test_split(
    features, target, test_size=0.2, stratify=target, random_state=2)

# Balance data using RandomOverSampler
ros = RandomOverSampler(sampling_strategy='minority', random_state=22)
X, Y = ros.fit_resample(X_train, Y_train)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_val = scaler.transform(X_val)

# Models
models = [
    LogisticRegression(),
    XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    SVC(kernel='rbf', probability=True)
]

# Training and evaluation
for model in models:
    model.fit(X, Y)
    print(f'{model} :')
    train_preds = model.predict_proba(X)
    print('Training ROC AUC:', metrics.roc_auc_score(Y, train_preds[:, 1]))
    val_preds = model.predict_proba(X_val)
    print('Validation ROC AUC:', metrics.roc_auc_score(Y_val, val_preds[:, 1]))
    print()

# Confusion Matrix for SVC
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(models[2], X_val, Y_val)
plt.title("SVC Confusion Matrix")
plt.show()

# Classification report
print("Classification Report (SVC):")
print(metrics.classification_report(Y_val, models[2].predict(X_val)))