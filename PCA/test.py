# STANDARDISE
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

sc = StandardScaler()
Z = sc.fit_transform(X)

print(X)

# Amount of PCs to keep
# Applying PCA
pca = PCA()
PCA_X = pca.fit_transform(Z)
explained_variance = pca.explained_variance_ratio_

print(PCA_X)

# -1.1619 - 1.1619 - 1.1619
# -0.3873 - 0.3873 - 0.3873
# 0.3873    0.3873   0.3873
# 1.1619
# 1.1619
# 1.1619
#
# 0.5774 - 0.5961 - 0.5580
# 0.5774 0.7813 - 0.2372
# 0.5774 - 0.1852 0.7952
