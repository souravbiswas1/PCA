# PCA

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('sample.csv')
data = dataset.iloc[:, 2:12]
data_corr = data.corr()
print(data_corr)
X = dataset.iloc[:, 2:12].values


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 5)
X_pca = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_
print('List of variance : ',explained_variance)
print('Reduced data : ',X_pca)