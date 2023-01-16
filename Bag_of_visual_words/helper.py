import tensorflow as tf
import cv2 as cv
import math
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Label	Description
# 0	T-shirt/top
# 1	Trouser
# 2	Pullover
# 3	Dress
# 4	Coat
# 5	Sandal
# 6	Shirt
# 7	Sneaker
# 8	Bag
# 9	Ankle boot

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

sift = cv.SIFT_create()

class_name = ['tshirt','trouser','pullover', 'dress','coat','sandal','shirt','sneaker','bag','ankle boot']



def euler_dist(d1,d2): 
  d = d1 - d2
  return np.linalg.norm(d)
  

def centroid(cluster): 
  cluster = np.array(cluster)
  cluster_centroid = np.mean(cluster,axis = 0)
  return cluster_centroid

def nearest_centroid(data,centroids):
  k = len(centroids)
  diff_array = []
  for i in range(k):
    diff_array.append(euler_dist(data,centroids[i]))
  diff_array = np.array(diff_array)
  return int(diff_array.argmin())


def extract_feature_for_each_image(ImageData):
  imageDescriptor = []
  length = len(ImageData)
  for i in range(length):
    img = ImageData[i]
    image8bit = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    k,descriptor = sift.detectAndCompute(image8bit,None)
    if descriptor is None: imageDescriptor.append([])
    else : imageDescriptor.append(list(descriptor))
  return imageDescriptor

def extractFeatures(descriptorSet):
  Descriptors = []
  length = len(descriptorSet)
  for i in range(length):
    descriptorLength = len(descriptorSet[i])
    for j in range(descriptorLength):
      Descriptors.append(descriptorSet[i][j])
  return np.array(Descriptors)



def k_Means(k,centroids,descriptors):
  n = len(descriptors)
  color = [0]*n
  for i in range(n):
    color[i] = nearest_centroid(descriptors[i],centroids)    
  newCentroids = []
  cluster = {}
  for i in range(k): cluster[i] = []
  for j in range(n): cluster[color[j]].append(descriptors[j])
  for i in range(k):
    newCentroids.append(centroid(cluster[i]))
  return (color,np.array(newCentroids))  

def initializeCentroid(k,descriptors):
      n = len(descriptors)
      x = list(np.random.choice(n,size = 1,replace = False))
      ret = []
      ret.append(descriptors[x[0],::])
      for i in range(k-1):
        D = [0]*n
        W = [0]*n
        total = 0
        for j in range(n):
          nearest = ret[nearest_centroid(descriptors[j,::],ret)]
          D[j] = euler_dist(descriptors[j,::],nearest)
          total += D[j]**2
        for j in range(n):
          W[j] = (D[j]**2)/total
        x = list(np.random.choice(n,size = 1,replace = False,p = W))
        ret.append(descriptors[x[0],::])
      #   print(ret)
      return np.array(ret)

def find_true_descriptors(centroids,descriptors):
  k = len(centroids)
  realRep = []
  for i in range(k):
    realRep.append(descriptors[nearest_centroid(centroids[i],descriptors)])
  return realRep




def computeHistogram(descriptors,clusterCentroids):
  k = len(clusterCentroids)
  descriptorNum=len(descriptors)
  frequencyVector=np.zeros(k)
  for j in range(descriptorNum):
    frequencyVector[nearest_centroid(descriptors[j],clusterCentroids)]+=1
  return frequencyVector


def getHistogram(clusterCentroids,image_descriptor):
  Histogram=[]
  length = len(image_descriptor)
  for i in range(length):
    descriptors = image_descriptor[i]
    Histogram.append(computeHistogram(descriptors,clusterCentroids))
  return np.array(Histogram)



def image_freq_for_words(histogramSet):
  k = histogramSet.shape[1]
  count = np.zeros(k)
  for histogram in histogramSet:
    for i in range(k):
      if histogram[i] > 0: count[i]+=1
  return count



def normalizeHistogramSet(histogramSet,image_num):
  normalized_histogram_set = []
  N = len(histogramSet)

  for Histogram in histogramSet:
    nd = 0
    k = len(Histogram)
    for i in range(k):
      if Histogram[i] > 0: nd+=1
    for i in range(k):
      if nd == 0 or image_num[i] == 0: Histogram[i] = 0
      else :Histogram[i] = (Histogram[i]/nd)*(math.log(N/image_num[i]))

    normalized_histogram_set.append(Histogram)

  return np.array(normalized_histogram_set)



def MatchHistogram(a,b):
  a = np.array(a)
  b = np.array(b)
  dot_prd = a*b
  mag_a = math.sqrt(np.sum(a**2))
  mag_b = math.sqrt(np.sum(b**2))

  if mag_a == 0:
    if mag_b == 0: return 0
    else: return 1
  elif mag_b == 0: return 1

  distance = dot_prd.sum()/(mag_a*mag_b)

  return (1 - distance)


def getIndex_of_Minimum(a):
  a = np.array(a)
  minpos = np.where(a == np.amin(a))
  minpos = minpos[0][0]
  return minpos

def getPredictions(train_label,test_labels,trainHistogramSet,testHistogramSet):
  n = len(trainHistogramSet)
  m = len(testHistogramSet)
  predictions = []

  for i in range(m):
    dis = []
    for j in range(n):
      dis.append(MatchHistogram(testHistogramSet[i],trainHistogramSet[j]))
    ind = getIndex_of_Minimum(dis)
    predictions.append(train_labels[ind])

  predictions = np.array(predictions)
  return predictions

def CreateVisualDictionary(k):
  print("Extracting Testing Images Data...")
  test_image_descriptor = extract_feature_for_each_image(test_images) 
  print("Extracting Training Images Data")
  train_image_descriptor = extract_feature_for_each_image(train_images)
  trainDescriptors = extractFeatures(train_image_descriptor)
  testDescriptors = extractFeatures(test_image_descriptor)
 
  print("On k = " + str(k))
  centroids = initializeCentroid(k,trainDescriptors)
  n = len(trainDescriptors)

  clusterCentroid = [0]*n
  for it in range(20): 
      print(datetime.now())
      print("Iteration " + str(it) + ":")
      (clusterCentroid,centroids) = k_Means(k,centroids,trainDescriptors)
      print(datetime.now())
  print("for k = " + str(k) + " the shape of the centroid is " + str(centroids.shape))
  centroids = find_true_descriptors(centroids,trainDescriptors)
  np.savetxt('./kmean31.txt',centroids)
  print("Forming histograms of training set.")
  trainHistogramSet = getHistogram(centroids,train_image_descriptor) #
  print("Forming histograms of testing set.")
  testHistogramSet = getHistogram(centroids,test_image_descriptor) #

  return trainHistogramSet,testHistogramSet