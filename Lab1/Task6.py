# import some packages we need

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Task 6.1
# Choose one image
colourful_face03 = cv2.imread('face 03 u6492108.jpg')

# resize it
face03_colorful = cv2.resize(colourful_face03, (512, 512))

def my_rotation(img, angle):
    '''
    input: 1. image to be rotated
           2. the degree of angle
    output:1. img1: forward warping image
            2. img2: backward warping image using nearest neighbour
            3  img3: backward warping image using bilinear interpolation
    '''
    img = img.astype(int)
    # Get old image h and w
    h, w = img.shape[0], img.shape[1]

    # Calculate the new h and w
    newW = int(w * abs(np.cos(angle)) + h * abs(np.sin(angle))) + 1
    newH = int(w * abs(np.sin(angle)) + h * abs(np.cos(angle))) + 1

    # Our three output images
    newimg1 = np.zeros((newH, newW, 3), dtype=int)
    newimg2 = np.zeros((newH, newW, 3), dtype=int)
    newimg3 = np.zeros((newH, newW, 3), dtype=int)

    # scr -> des
    trans1 = np.array([[1, 0, 0], [0, -1, 0], [-0.5 * w, 0.5 * h, 1]])
    trans1 = trans1 @ np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    trans1 = trans1 @ np.array([[1, 0, 0], [0, -1, 0], [0.5 * newW, 0.5 * newH, 1]])
    # des -> src
    trans2 = np.array([[1, 0, 0], [0, -1, 0], [-0.5 * newW, 0.5 * newH, 1]])
    trans2 = trans2 @ np.array([[np.cos(angle), np.sin(angle), 0], [-np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    trans2 = trans2 @ np.array([[1, 0, 0], [0, -1, 0], [0.5 * w, 0.5 * h, 1]])

    # Forward Warping
    for x in range(w):
        for y in range(h):
            # Find the position in target image
            newPos = np.array([x, y, 1]) @ trans1
            # move the greyscale values to target image
            # If the new position is not on the pixel, we round it to the nearest pixel
            newimg1[int(newPos[1])][int(newPos[0])] = img[y][x]

    # Backward Warping
    for x in range(newW):
        for y in range(newH):
            # Get the position in the source image
            srcPos = np.array([x, y, 1]) @ trans2
            # Check if the position is in the range of the source image
            if srcPos[0] >= 0 and srcPos[0] < w and srcPos[1] >= 0 and srcPos[1] < h:
                # nearest neighbour
                newimg2[y][x] = img[int(srcPos[1])][int(srcPos[0])]
                # biliear
                bix, biy = int(srcPos[0]), int(srcPos[1])  # 取左上角坐标
                if bix < w - 1 and biy < h - 1:
                    rgbX1 = img[biy][bix] + (img[biy][bix + 1] - img[biy][bix]) * (srcPos[0] - bix)
                    rgbX2 = img[biy + 1][bix] + (img[biy + 1][bix + 1] - img[biy + 1][bix]) * (srcPos[0] - bix)
                    rgb = rgbX1 + (rgbX2 - rgbX1) * (srcPos[1] - biy)
                    newimg3[y][x] = rgb

    return newimg1, newimg2, newimg3

# 6.1
# -90 , -45 , -15 , 45 , and 90
fig = plt.figure(figsize=(14, 10))

# swap channels
face03_colorful[:, :, [0, 2]] = face03_colorful[:, :, [2, 0]]

# -90 rotation
img1, img2, img3 = my_rotation(face03_colorful, -90 * np.pi/180)

img1 = np.uint8(img1)
img2 = np.uint8(img2)
img3 = np.uint8(img3)

ax1 = fig.add_subplot(231)
ax1.imshow(img3)
ax1.set_title('face03 with -90 degree rotation')

# -45
img1, img2, img3 = my_rotation(face03_colorful, -45 * np.pi/180)

img1 = np.uint8(img1)
img2 = np.uint8(img2)
img3 = np.uint8(img3)

ax2 = fig.add_subplot(232)
ax2.imshow(img3)
ax2.set_title('face03 with -45 degree rotation')

# -15
img1, img2, img3 = my_rotation(face03_colorful, -15 * np.pi/180)

img1 = np.uint8(img1)
img2 = np.uint8(img2)
img3 = np.uint8(img3)

ax3 = fig.add_subplot(233)
ax3.imshow(img3)
ax3.set_title('face03 with -15 degree rotation')

# 45
img1, img2, img3 = my_rotation(face03_colorful, 45 * np.pi/180)

img1 = np.uint8(img1)
img2 = np.uint8(img2)
img3 = np.uint8(img3)

ax4 = fig.add_subplot(234)
ax4.imshow(img3)
ax4.set_title('face03 with 45 degree rotation')

# 90
img1, img2, img3 = my_rotation(face03_colorful, 90 * np.pi/180)

img1 = np.uint8(img1)
img2 = np.uint8(img2)
img3 = np.uint8(img3)

ax5 = fig.add_subplot(235)
ax5.imshow(img3)
ax5.set_title('face03 with 90 degree rotation')

#4.2
# compare forward and backward warping
# rotate -15 degrees
img1, img2, img3 = my_rotation(face03_colorful, -15 * np.pi/180)

img1 = np.uint8(img1)
img2 = np.uint8(img2)
img3 = np.uint8(img3)

# Plot
fig = plt.figure(figsize=(10, 7))

ax1 = fig.add_subplot(121)
ax1.imshow(img1)
ax1.set_title('Forward warping')
ax1 = fig.add_subplot(122)
ax1.imshow(img3)
ax1.set_title('Backward warping')

#4.3

# Plot
fig = plt.figure(figsize=(10, 7))

ax1 = fig.add_subplot(121)
ax1.imshow(img2)
ax1.set_title('Nearest neighbour')
ax1 = fig.add_subplot(122)
ax1.imshow(img3)
ax1.set_title('Bilinear interpolation')

plt.show()
cv2.waitKey(0)