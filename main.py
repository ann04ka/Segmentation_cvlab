import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from k_means import KMeans
import random

def getClassesByHist(n, img):
    """
    Segments grayscale image into n equal areas

    Parameters:
    - n (int): number of classes
    - img (array): original grayscale image

    Returns:
    - result (array): image after processing
    """
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    mean_areas = (img.shape[0] * img.shape[1])/n
    light_list = []
    summ = 0

    light_list.append(0)
    for i in range(256):
        summ += hist[i]
        if summ >= mean_areas:
            light_list.append(i)
            summ = 0

    result = np.ndarray((img.shape[0], img.shape[1]), dtype=np.uint8)
    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            for i in range(len(light_list)):
                if img[h][w] >= light_list[i]:
                    result[h][w] = int(255 / (n-1)) * i
    # vals = result.flatten()
    # b, bins, patches = plt.hist(vals, 255)
    # plt.xlim([0, 255])
    # plt.show()

    return(result)

def elbow_method(X, max_k = 10):
    costs = []
    for k in range(2, max_k):
        model = KMeans(n_clusters=k)
        model.fit(X)
        costs.append(model.cost_)

    plt.close()
    plt.plot(list(range(2, max_k)), costs)
    plt.xlabel("# of clusters (K)")
    plt.ylabel("Cost")
    plt.show()

# getClassesByHist(n, img)
# elbow_method(input)

input = cv.imread('varan.jpg', cv.IMREAD_GRAYSCALE)
print(input)
# input = cv.cvtColor(input, cv.COLOR_BGR2GRAY)
input_shape = input.shape
input = input / 255
input = input.reshape(-1, 1)
#
model = KMeans(n_clusters=3)
cluster_means, image_data_with_clusters = model.fit(input)
output = np.zeros(input.shape)

for i, cluster in enumerate(image_data_with_clusters[:, -1]):
    output[i, :] = cluster_means[int(cluster)]

output_reshaped = output.reshape(input_shape)

plt.close()
plt.axis('off')
plt.imshow(output_reshaped, cmap="gray")
plt.show()

