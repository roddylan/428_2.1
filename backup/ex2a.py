import numpy as np
import matplotlib.pyplot as plt
import cv2


def load_img(f):
    img = cv2.imread(f)

    return img

def load_gimg(f):
    img = load_img(f)

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



img = load_img("football_field.jpg")


dim = [img.shape[1], img.shape[0]]
xs = np.arange(0, dim[0], dtype=np.float64)
ys = np.arange(0, dim[1], dtype=np.float64)

plt.imshow(img)
plt.title("select 4 corners (p1-p2 = bot. L - top R, p3-p4 = bot. R - top L)")
points = plt.ginput(4, timeout=0)
plt.close()
# plt.show()
hps = []


for (x,y) in points:
    p = np.array([x, y, 1]).reshape(3,1)
    hps.append(p)


hp1, hp2, hp3, hp4 = hps



l1 = np.cross(hp1.T, hp2.T).T
l2 = np.cross(hp3.T, hp4.T).T
pm = np.cross(l1.T, l2.T).T

# parallel line
l3 = np.cross(hp1.T, hp4.T).T
l4 = np.cross(hp2.T, hp3.T).T

p_inf = np.cross(l3.T,l4.T).T

lm = np.cross(p_inf.T, pm.T).T

# ax + by + c = 0 -> by = -ax - c -> y = (-ax - c)/b -> y = -(ax + c)/b


l1_ys = np.array([], dtype=np.float64)
l2_ys = np.array([], dtype=np.float64)
l3_ys = np.array([], dtype=np.float64)
l4_ys = np.array([], dtype=np.float64)
lm_ys = np.array([], dtype=np.float64)

# print(l1.shape, l2.shape, l3.shape, l4.shape, lm.shape)
for x in xs:
    # a = [0]; b = [1]; c = [2]
    l1_y = -(l1[2] + l1[0] * x)/l1[1]
    l2_y = -(l2[2] + l2[0] * x)/l2[1]
    l3_y = -(l3[2] + l3[0] * x)/l3[1]
    l4_y = -(l4[2] + l4[0] * x)/l4[1]
    lm_y = -(lm[2] + lm[0] * x)/lm[1]
    
    l1_ys = np.append(l1_ys, l1_y)
    l2_ys = np.append(l2_ys, l2_y)
    l3_ys = np.append(l3_ys, l3_y)
    l4_ys = np.append(l4_ys, l4_y)
    lm_ys = np.append(lm_ys, lm_y)

# fig = plt.figure()
# ax = plt.axes()

# ax.imshow(img[:,:,::-1])
# ax.plot(l1_ys)
# ax.plot(l2_ys)
# ax.plot(l3_ys)
# ax.plot(l4_ys)
# ax.plot(lm_ys)

p1 = np.array([hp1[0]/hp1[2], hp1[1]/hp1[2]])
p2 = np.array([hp2[0]/hp2[2], hp2[1]/hp2[2]])
p3 = np.array([hp3[0]/hp3[2], hp3[1]/hp3[2]])
p4 = np.array([hp4[0]/hp4[2], hp4[1]/hp4[2]])

pmx, pmy = pm[0]/pm[2], pm[1]/pm[2]
pxs = [p1[0], p2[0], p3[0], p4[0], pmx]
pys = [p1[1], p2[1], p3[1], p4[1], pmy]

fig,ax = plt.subplots()
ax.imshow(img[:,:,::-1])

ax.set_xlim(left=0, right=dim[0])
ax.set_ylim(bottom=dim[1], top=0)

# ax.plot(xs, l1_ys)
# ax.plot(xs, l2_ys)
ax.plot(xs, l3_ys)
ax.plot(xs, l4_ys)
ax.plot(xs, lm_ys)

# plt.show()
ax.scatter(pxs,pys)

txt = ['p1', 'p2', 'p3', 'p4', "mid"]
print(p1)
print(p2)
print(p3)
print(p4)
for i in range(len(txt)):
    ax.annotate(txt[i], (pxs[i], pys[i]))
plt.show()



# plt.show()

    

