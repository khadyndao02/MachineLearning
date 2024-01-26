import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA


X,y = make_blobs(n_samples = 100, centers = 3, cluster_std=0.5)
print(X)
print(y)
plt.scatter(X[:,0],X[:,1])
plt.show()
model = KMeans(init='k-means++', n_clusters=3, n_init=10)
print(model)
print(model.fit(X))
print(model.predict(X))
plt.scatter(X[:,0],X[:,1], c = model.predict(X))
plt.show()
print(model.cluster_centers_)
plt.scatter(X[:,0],X[:,1], c = model.predict(X))
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],c = 'r')
plt.show()
print(model.inertia_)
inertia = []
k_range = range(1,20)
for k in k_range:
    model = KMeans(n_clusters = k).fit(X)
    inertia.append(model.inertia_)
plt.plot(k_range, inertia)
plt.xlabel('nombres de clusters')
plt.ylabel('cout du modele')
plt.show()

#Détection d'anomalie

# IsolationForest
model = IsolationForest(contamination = 0.01)
print(model)
print(model.fit(X))
plt.scatter(X[:,0],X[:,1], c = model.predict(X))
plt.show()


digits = load_digits()
images = digits.images
X = digits.data
y = digits.target
print(X.shape)
plt.imshow(images[42])
plt.show()


model= IsolationForest(random_state =0, contamination = 0.02)
print(model)
print(model.fit(X))
print(model.predict(X))
plt.scatter(X[:,0],X[:,1], c = model.predict(X))
plt.show()
outliers = model.predict(X) == -1
print(outliers)
print(images[outliers])
print(images[outliers][0])
plt.imshow(images[outliers][0])
plt.title(y[outliers][0])
plt.show()

#réduction de dimension
# analyse en composantes principales

print(X.shape)
model = PCA(n_components = 2)
print(model)
X_reduced = model.fit_transform(X)
plt.scatter(X_reduced[:, 0],X_reduced[:, 1])
plt.scatter(X_reduced[:, 0],X_reduced[:, 1], c =y)
plt.colorbar()
plt.show()

print(model.components_)
print(model.explained_variance_ratio_)
print(np.cumsum(model.explained_variance_ratio_))
plt.plot(np.cumsum(model.explained_variance_ratio_))
plt.show()
