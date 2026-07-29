import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

x, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
df = pd.DataFrame(x, columns=["Feature1", "Feature2"])
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(df)
print(df.head(10))
