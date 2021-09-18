#import packages
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import glob
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")

#folder \Bilder is a copy of \qBilder (since they are downsized to 20x20 pixel)
#folder \qBilder contains original images

# pixels in an image and their distances to eachother in deltaE_ciede2000
class pixel_distances_within_image():
    def __init__(self,pic):
        self.pic=pic
    def smallify_image(self,n):
        imgs = Image.open(self.pic) 
        imgs.thumbnail((n, n))# n=20, m=20
        return imgs.save(self.pic)
    def reshape(self):
        image = cv2.imread(self.pic)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image = image.reshape((image.shape[0] * image.shape[1], 3))
        return image
    def transform2lab(self):
        return rgb2lab(self.reshape())
    def distances_between_pixel_df(self):
        dislab=pd.DataFrame()
        image=self.transform2lab()
        for i in range(0,len(image)-1):
            l = [0] * (i+1)
            lo=pd.DataFrame()
            for j in range(i+1,len(image)):
                data=deltaE_ciede2000(image[i],image[j])
                l.append(data)
            lo=lo.append(l).T
            dislab=dislab.append(lo).reset_index(drop=True)
        dislabT=dislab.T.reset_index(drop=True)
        distance_df=dislab.reset_index(drop=True).add(dislabT, fill_value=0).fillna(0)
        return distance_df

#calculate dominant colour clusters with KMedoids
class dominant_colours_via_KMedoids(): 
    def __init__(self, df):
        self.distance_df=df
    def kmedoids_labels(self,n_clusters=5):
        kmedoids = KMedoids(n_clusters,metric='precomputed').fit(self.distance_df)
        labels = kmedoids.predict(self.distance_df)
        #cc=kmedoids.cluster_centers_
        return labels
    def cluster(self, image, n_clusters=5):
        labels=self.kmedoids_labels()
        dist=self.distance_df
        dist.insert(0, 'Cluster Labels', labels)
        centroids=[]
        for n in range(0,n_clusters):
            cluster=dist.loc[dist['Cluster Labels'] == n][dist.index[dist['Cluster Labels'] == n].tolist()]
            summe=cluster.sum(axis=1)
            cluster['sum']=summe
            centroid_row=cluster.loc[cluster['sum']==min(summe)]
            centroid=centroid_row.index[0]
            centroids.append(image[centroid])
        return labels, centroids
    
#Visualisation of dominant colours
class visualise_dominant_colours():
    def __init__(self, labels, centroids):
        self.labels=labels
        self.centroids=centroids
    def centroid_histogram(self):
        numLabels = np.arange(0, len(np.unique(self.labels)) + 1)
        (hist, _) = np.histogram(self.labels, bins = numLabels)
        hist = hist.astype("float")
        hist /= hist.sum()
        return hist
    def plot_colours(self):
        bar = np.zeros((50, 300, 3), dtype = "uint8")
        startX = 0
        for (percent, color) in zip(self.centroid_histogram(), self.centroids):
            endX = startX + (percent * 300)
            cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                color.astype("uint8").tolist(), -1)
            startX = endX
        return bar
    def plot_and_save_dominant_colours(self,name):
        bar = self.plot_colours()
        plt.figure()
        plt.axis("off")
        plt.imshow(bar)
        plt.savefig(name,dpi=400,bbox_inches='tight')
        plt.show()
    
    #transform each picture to a 1x100 image of its cluster colours
    def simplify_colours(self):
        bar = np.zeros((1, 100, 3), dtype = "uint8")
        startX = 0
        for (percent, color) in zip(self.centroid_histogram(), self.centroids):
            endX = startX + (percent * 100)
            cv2.rectangle(bar, (int(startX), 0), (int(endX), 1),
                color.astype("uint8").tolist(), -1)
            startX = endX
        return bar

#import os

# detect the current working directory and print it
#path = os.chdir(os.getcwd()+'\Documents\GitHub\Playing_with_data')
#try:
#    os.mkdir(path+"\boxBilder")
#    os.mkdir(path+"\clusterBilder")
#except OSError:
#    print ("Creation of directories failed")
#else:
#    print ("Successfully created the directories")


image_names=glob.glob("Bilder/*.jpg")

#scale down images too 20x20 pixels
for i in image_names:
    pixel_distances_within_image(i).smallify_image(20)
 
images_colours=[]    

for i in image_names:
    bsp=pixel_distances_within_image(i)
    dom=bsp.distances_between_pixel_df()
    km=dominant_colours_via_KMedoids(dom)
    clstr=km.cluster(bsp.reshape())
    vis=visualise_dominant_colours(clstr[0], clstr[1])
    #vis.plot_and_save_dominant_colours("box"+i)
    ic=vis.simplify_colours()
    images_colours.append(ic)
    
    
#transform to lab
lab_images=[]
for image in images_colours:
    lab_img=rgb2lab(image)
    lab_images.append(lab_img)

#lab=lab_images[n]
def distance_vec(lab1,lab2):
    dist=[]
    for pxl1 in range(0,len(np.unique(lab1[0], axis=0, return_counts = True)[0])):
        idis=[]
        for pxl2 in range(0,len(np.unique(lab2[0], axis=0, return_counts = True)[0])):
            d=deltaE_ciede2000(np.unique(lab1[0], axis=0, return_counts = True)[0][pxl1], np.unique(lab2[0], axis=0, return_counts = True)[0][pxl2])
            idis.append(d)
        dist.append(min(idis))
    return dist 

similarity_threshold = 5

def similarity_vec(lab1,lab2):
    sim_vec=[]
    for d in distance_vec(lab1,lab2):
        if d<similarity_threshold:
            sim_vec.append(1)
        else:
            sim_vec.append(0)
    return sim_vec   


data=[]
for i in range(0,len(lab_images)):
    l=[]
    for j in range(0,len(lab_images)):
        sm=sum(similarity_vec(lab_images[i],lab_images[j]))
        l.append(sm)
    data.append(l)
    
sim=pd.DataFrame(data=data,index=image_names, columns=image_names)
kmeans = KMeans(n_clusters=20, random_state=0).fit(sim)
sim.insert(0, 'Cluster Labels', kmeans.labels_)

def show_cluster_with_box(cluster_label):
    for i in range(0,len(sim)):
        if sim["Cluster Labels"][i]==cluster_label:
            print("Cluster "+str(cluster_label)+": "+sim.index[i])
            f, axarr = plt.subplots(1,2)
            axarr[0].imshow(cv2.cvtColor(cv2.imread('q'+sim.index[i]), cv2.COLOR_BGR2RGB))
            axarr[0].axis('off')
            #plt.savefig('cluster'+sim.index[i].rsplit('\\')[0]+'\\'+str(i)+sim.index[i].rsplit('\\')[1],dpi=400,bbox_inches='tight')
            axarr[1].imshow(cv2.cvtColor(cv2.imread("box"+sim.index[i]), cv2.COLOR_BGR2RGB))
            axarr[1].axis('off')

def show_cluster(cluster_label):
    for i in range(0,len(sim)):
        if sim["Cluster Labels"][i]==cluster_label:
            print("Cluster "+str(cluster_label)+": "+sim.index[i])
            plt.figure()
            plt.axis('off')
            plt.imshow(cv2.cvtColor(cv2.imread('q'+sim.index[i]), cv2.COLOR_BGR2RGB))
            plt.savefig('cluster'+sim.index[i].rsplit('\\')[0]+'\\CLUSTER'+str(cluster_label)+"_"+sim.index[i].rsplit('\\')[1],dpi=400,bbox_inches='tight')
            plt.show()
            
for i in np.unique(kmeans.labels_).tolist():
    print('\n---------------------------------------------\n')
    print("Images of cluster "+str(i))
    print('\n---------------------------------------------\n')
    show_cluster(i)