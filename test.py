import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import cv2, os
import shutil # for folders




img1 = cv2.imread("3_in/key2.jpg")

plt.imshow(img1)
pts = plt.ginput(4,timeout=0)
plt.close()



hps = [np.array([pt[0], pt[1], 1]).reshape(-1,) for pt in pts]


A = np.empty((0,3),dtype=np.float32)
for hp in hps:
    A = np.vstack((A, hp))

xs = np.array([hp[0] for hp in hps])
ys = np.array([hp[1] for hp in hps])

# print(A.shape)
# c = np.mean(A,axis=0)
w = np.max(xs) - np.min(xs)
h = np.max(ys) - np.min(ys)

# print(w,h)

T = np.linalg.inv(np.array([
    [w+h, 0, w/2],
    [0, w+h, h/2],
    [0, 0, 1]
]))

# T_inv = np.array([
#     [w+h, 0, w/2],
#     [0, w+h, h/2],
#     [0, 0, 1]
# ])

print(T)
o = np.array([0,0,1]).reshape(3,1)
# print(c.shape)
# print(c)
c = T @ o
print(c)

px = []
py = []

for pt in hps:
    p = T @ pt
    # p = pt
    # print(p[0])
    px.append(p[0].reshape(1,-1))
    py.append(p[1].reshape(1,-1))

px.append(c[0].reshape(1,-1))
py.append(c[1].reshape(1,-1))

plt.scatter(px, py)
plt.annotate("centroid", (c[0],c[1]))
plt.show()