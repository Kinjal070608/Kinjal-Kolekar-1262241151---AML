import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
data = pd.read_csv(url, header=None)
X = data.iloc[:, 2:]
y = data.iloc[:, 1]
y = y.map({'M': 1, 'B': 0})
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42
)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Actual Values:")
print(y_test.values)
print("\nPredicted Values:")
print(y_pred)
print("\nAccuracy =", accuracy * 100, "%")
