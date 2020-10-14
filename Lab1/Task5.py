# import some packages we need

import numpy as np
import cv2
import matplotlib.pyplot as plt

colourful_face03 = cv2.imread('face 03 u6492108.jpg')

# Get the grey version of it
face03_grey  = colourful_face03[:, :, 0] * 0.11 + colourful_face03[:, :, 1] * 0.59 + colourful_face03[:, :, 2] * 0.3
face03_grey = face03_grey.astype(np.uint8)


def Sobel_Kernel():
    '''
    This function will output two Sobel matrix (one is vertical and one is horizontal)
    '''
    sobel_V = np.array(([1, 0, -1], [2, 0, -2], [1, 0, -1]))
    sobel_H = np.array(([1, 2, 1], [0, 0, 0], [-1, -2, -1]))
    return sobel_V, sobel_H

def Sobel(image, kernel):
    '''
    input: image
    '''
    rows = image.shape[0]
    cols = image.shape[1]
    # The sobel operator is always 3x3, so the radius is 1
    rad = 1

    temp_img =  np.zeros((rows + 2 * rad, cols + 2 * rad))
    # Fill up the temp image matrix
    for i in range(rows):
        for j in range(cols):
            temp_img[i + 1, j + 1] = image[i, j]

    img = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            img[i,j] = np.sum(temp_img[i:i+3, j:j+3] * kernel)

    return img

fig = plt.figure(figsize=(14, 10))

sobel_v, sobel_h = Sobel_Kernel()

Sobel_V = Sobel(face03_grey, sobel_v)
Sobel_H = Sobel(face03_grey, sobel_h)
Sobel_full = np.sqrt(Sobel_V ** 2 + Sobel_H ** 2)

Sobel_V = cv2.convertScaleAbs(Sobel_V)
Sobel_H = cv2.convertScaleAbs(Sobel_H)


Sobel_full = np.uint8(Sobel_full)
Sobel_full = cv2.convertScaleAbs(Sobel_full)

ax1 = fig.add_subplot(231)
ax1.imshow(Sobel_V, cmap = 'gray')
ax1.set_title('My Sobel_V')

ax2 = fig.add_subplot(232)
ax2.imshow(Sobel_H, cmap = 'gray')
ax2.set_title('My Sobel_H')

ax3 = fig.add_subplot(233)
ax3.imshow(Sobel_full, cmap = 'gray')
ax3.set_title('My Sobel_full')

# Using built-in functions to apply Sobel
edges = cv2.Sobel(face03_grey, cv2.CV_16S, 1, 0, ksize = 3)
edges = cv2.convertScaleAbs(edges)

ax4 = fig.add_subplot(234)
ax4.imshow(edges, cmap = 'gray')
ax4.set_title('Built-in Sobel_V')

edges = cv2.Sobel(face03_grey, cv2.CV_16S, 0, 1, ksize = 3)
edges = cv2.convertScaleAbs(edges)

ax5 = fig.add_subplot(235)
ax5.imshow(edges, cmap = 'gray')
ax5.set_title('Built-in Sobel_H')

edges = cv2.Sobel(face03_grey, cv2.CV_16S, 1, 1, ksize = 3)
edges = cv2.convertScaleAbs(edges)

ax6 = fig.add_subplot(236)
ax6.imshow(edges, cmap = 'gray')
ax6.set_title('Built-in Sobel_full')

plt.show()
