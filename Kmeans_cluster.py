from pandas import DataFrame #get data
import matplotlib
import matplotlib.pyplot as plt # to plot the visualization
from sklearn.cluster import KMeans # to use the Kmean method form Sklearn library

Data = {'x': [25,34,22,27,33,33,31,22,35,34,67,54,57,43,50,57,59,52,65,47,49,48,35,33,44,45,38,43,51,46],
        'y': [79,51,53,78,59,74,73,57,69,75,51,32,40,47,53,36,35,58,59,50,25,20,14,12,20,5,29,27,8,7]
       }
  
def k_means(cluster,df):
	kmeans = KMeans(n_clusters=cluster).fit(df)
	centroids = kmeans.cluster_centers_
	for i in range(len(centroids)):
		print('the centroid {} coordinates {}'.format(i+1, centroids[i]))
	plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50)
	plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
	return kmeans
def prediction(x,y,kmeans):
	cluster_name = kmeans.predict([[x,y]])
	return cluster_name
def main():
	x_test= 1
	y_test= 1
	clusters=4
	df = DataFrame(Data,columns=['x','y'])
	Kmean=k_means(clusters,df)
	cluster = prediction(x_test,y_test,Kmean)
	print ( '[{},{}] belongs to cluster {}'.format(x_test,y_test,cluster))
if __name__ == '__main__':
	main()