## importing the file
import numpy as np
import pandas as pd
data=pd.read_csv('E:\\assignment\\pca\\wine.csv')
data.info()
data.head()



## Normalizing the numerical data 
data_new=(data-data.min())/(data.max()-data.min())
data_new.info()
data_new.describe()


## Creating a PCA instance and fitting it to the data
from sklearn.decomposition import PCA
pca = PCA(n_components=14)
principalComponents = pca.fit_transform(data_new)


## variance that each PCA explains is 
var = pca.explained_variance_ratio_
var
pca.components_[0]

# Cumulative variance 
var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1

## Variance plot for PCA components obtained
import matplotlib.pyplot as plt 
plt.plot(var1,color="b")


# Save components to a DataFrame
PCA_components = pd.DataFrame(principalComponents[:,:3])


##clustering using first 3 compoonents
from sklearn.cluster import KMeans

##scree plot to find the nummber of clusters
twss =[]
for i in range(1, 10):
    kmeans = KMeans(n_clusters = i).fit(PCA_components)
    kmeans.fit(PCA_components)
    twss.append(kmeans.inertia_)
    
    
##ploting the data   

plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('twss')
plt.plot(range(1, 10), twss)
plt.show()
### from the scree plot we got cluster =3

 
kmeans = KMeans(n_clusters = 3)
kmeans.fit(PCA_components)
kmeans.labels_
K_clusters= pd.Series(kmeans.labels_)

data['cluster_K']=K_clusters
data.head()


## heirarchical clustering
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 10))  
plt.title("Dendrograms")  
dendogram = shc.dendrogram(shc.linkage(PCA_components))
plt.axhline(y=0.3, color='r', linestyle='--')


## As we have3 clusters, now appling heirarchical clustering 
from sklearn.cluster import AgglomerativeClustering
heirarchical = AgglomerativeClustering(n_clusters=3, affinity='euclidean').fit(PCA_components)  
heirarchical.labels_


heirarchical= pd.Series(heirarchical.labels_)

### inserting new column in the table
data['clusters_h']=heirarchical
data.head()
