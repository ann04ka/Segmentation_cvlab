# Квантование и кластеризация изображений

## Обзор
Этот проект включает реализацию различных алгоритмов для квантования и кластеризации изображений. Включены задачи по квантованию монохромного изображения в n-градаций, реализация k-means для монохромного и цветного изображений, а также метод локтя для определения оптимального количества кластеров.
**Оригинальное изображение** ![Оригинальное изображение](https://github.com/ann04ka/Segmentation_cvlab/blob/1/varan.jpg)



## Задачи

### Задача 1(а): Квантование монохромного изображения
Реализовать алгоритм, квантующий монохромное изображение в n-градаций.
**Пример** ![Пример](https://github.com/ann04ka/Segmentation_cvlab/blob/1/11.png)

#### Функция
```python
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

    return result
```
**Квантование по гистаграмме** ![Гист](https://github.com/ann04ka/Segmentation_cvlab/blob/cfc45ed6ecf1ef12125b54a398f6bc171543f5e9/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA1.PNG)

### Задача 1(б): k-means для монохромного изображения
Реализовать k-means для монохромного изображения.
**Пример** ![Пример](https://github.com/ann04ka/Segmentation_cvlab/blob/1/1.png)

#### Пример кода
```python
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

input = cv.imread('varan.jpg')
input = cv.cvtColor(input, cv.COLOR_BGR2GRAY)
input_shape = input.shape
input = input / 255
input = input.reshape(-1, 1)

model = KMeans(n_clusters=5)
cluster_means, image_data_with_clusters = model.fit(input)

output = np.zeros(input.shape)
for i, cluster in enumerate(image_data_with_clusters[:, -1]):
    output[i, :] = cluster_means[int(cluster)]

output_reshaped = output.reshape(input_shape)

plt.close()
plt.axis('off')
plt.imshow(output_reshaped, cmap="gray")
plt.show()
```

### Задача 2: k-means для цветного изображения
Реализовать k-means для цветного изображения.
**Пример** ![Пример](https://github.com/ann04ka/Segmentation_cvlab/blob/1/2.png)

#### Пример кода
```python
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

input = cv.imread('varan.jpg')
input = cv.cvtColor(input, cv.COLOR_BGR2RGB)
input_shape = input.shape
input = input / 255
input = input.reshape(-1, 3)

model = KMeans(n_clusters=5)
cluster_means, image_data_with_clusters = model.fit(input)

output = np.zeros(input.shape)
for i, cluster in enumerate(image_data_with_clusters[:, -1]):
    output[i, :] = cluster_means[int(cluster)]

output_reshaped = output.reshape(input_shape)

plt.close()
plt.axis('off')
plt.imshow(output_reshaped)
plt.show()
```

### Задача 3: Метод локтя для определения количества кластеров
Реализовать метод локтя для определения оптимального количества кластеров.
**Пример** ![Пример](https://github.com/ann04ka/Segmentation_cvlab/blob/1/3.png)

#### Функция
```python
def elbow_method(X, max_k=10):
    costs = []
    for k in range(2, max_k):
        model = KMeans(n_clusters=k)
        model.fit(X)
        costs.append(model.inertia_)

    plt.close()
    plt.plot(list(range(2, max_k)), costs)
    plt.xlabel("# of clusters (K)")
    plt.ylabel("Cost")
    plt.show()
```
**Сравнение всех методов при разном числе классов** ![методы](https://github.com/ann04ka/Segmentation_cvlab/blob/cfc45ed6ecf1ef12125b54a398f6bc171543f5e9/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA2.PNG)
