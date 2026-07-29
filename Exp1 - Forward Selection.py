import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector

iris = load_iris()
x = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
model = LinearRegression()
sfs = SequentialFeatureSelector(model, n_features_to_select=2, direction="forward", scoring="r2", cv=5)
sfs.fit(x, y)
selected_features = x.columns[sfs.get_support()]
print("Selected Features:")
print(selected_features.tolist())
