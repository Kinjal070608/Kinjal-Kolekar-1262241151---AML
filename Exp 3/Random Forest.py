import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
accuracy_score,
confusion_matrix,
classification_report,
ConfusionMatrixDisplay
)

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")
print("Dataset loaded successfully.")
print("\nDataset Shape:")
print(X.shape)
print("\nFirst five records:")
print(X.head())
print("\nClass names:")
print(data.target_names)
print("\nClass Distribution:")
print(y.value_counts())
X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.20,
random_state=42,
stratify=y
)
print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
rf_model = RandomForestClassifier(
n_estimators=100,
random_state=42
)
rf_model.fit(X_train, y_train)
print("\nRandom Forest model trained successfully.")
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nTest Accuracy:")
print(f"{accuracy * 100:.2f}%")
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(
classification_report(
y_test,
y_pred,
target_names=data.target_names
)
)
disp = ConfusionMatrixDisplay(
confusion_matrix=cm,
display_labels=data.target_names
)
disp.plot()
plt.title("Random Forest - Confusion Matrix")
plt.show()
