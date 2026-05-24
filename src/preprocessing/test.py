import os
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
import joblib

L_DIR = "test_dataset/L_luminance"
Y_DIR = "test_dataset/class_labels"

KMEANS_PATH = "test_kmeans.pkl"

files = os.listdir(L_DIR)
file_name = random.choice(files)

print("Wybrany plik:", file_name)

L = np.load(os.path.join(L_DIR, file_name))
labels = np.load(os.path.join(Y_DIR, file_name))

print("L shape:", L.shape)
print("labels shape:", labels.shape)

kmeans = joblib.load(KMEANS_PATH)

ab = kmeans.cluster_centers_[labels]


lab = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.float32)

lab[:, :, 0] = L
lab[:, :, 1:3] = ab

rgb = lab2rgb(lab)

plt.figure(figsize=(8, 8))
plt.imshow(rgb)
plt.title(file_name)
plt.axis("off")
plt.show()