# Aesthetics

 **Aesthetics.py**: <br />
 Python script which takes the jpg-images in a folder "Bilder" and clusters them according to their similarity in colour into a new folder "clusterBilder". "OriginalBilder" contains the original images since they are sized down in the process.<br />

  **Goal**: Order images by similarity of their dominant colours for aesthetics.<br />
  **Method**: Finding the dominant colours in images using a clustering algorithm, calculating distances in Cielab colour space using the metric Cielab ΔE* CIEDE2000 to account for distances as perceived by the human eye. Second KMeans clustering to find clusters of images which look aesthetically together colourwise.<br />
    The left side shows the original image, while the right depicts the dominant colours of the corresponding image clustered with KMeans and RGB. One can see that the dominant colours in the last image are particularly off, since perception of lighting is a major factor for using CIEDE2000.<br />
  ![alt text](https://github.com/Kokostino/Aesthetics/blob/main/files/cluster1.PNG?raw=true)<br />
  In **Aesthetics.ipynb**, KMeans and RGB coordinates (having a euclidean metric) is used for finding the dominant colours in an image. Ideally one'd use Lab space and ΔE*-metric. In **Aesthetics_KMedoids.ipynb** we use KMediods for clustering which also works for non-euclidean distances. KMedoids is similar to KMeans but instead of calculating the mean to set the centroid position, the most central data point itsself is set as centroid. As input, the distances of pixels in CIEDE2000 are sufficient.<br />
  The upper image shows dominant colours found via KMedoids, the lower one via KMean. The clustered image is the last one from the previous plot, where the colours by KMeans were a bit off.<br />
  ![alt text](https://github.com/Kokostino/Aesthetics/blob/main/files/MedvsMean.PNG)<br />
 

