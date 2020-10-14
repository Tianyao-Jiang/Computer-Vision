# import some packages we need

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Task 3.3
# a

# Create a figure
fig = plt.figure(figsize=(14, 10))

# Read this face image from its JPG file,
face03 = cv2.imread('face 03 u6492108.jpg')

# resize the image to 768 x 512 in columns x rows
face03 = cv2.resize(face03, (768, 512), interpolation = cv2.INTER_AREA)
# Swap the R and B channels
face03[:, :, [0, 2]] = face03[:, :, [2, 0]]
# plot
ax1 = fig.add_subplot(241)
ax1.imshow(face03)
ax1.set_title('face 03')

# b
R = face03[:, :, 0]
G = face03[:, :, 1]
B = face03[:, :, 2]

# For R channel
ax2 = fig.add_subplot(242)
ax2.imshow(R, cmap='gray')
ax2.set_title('R channel grayscale image')

# For G channel
ax3 = fig.add_subplot(243)
ax3.imshow(G, cmap='gray')
ax3.set_title('G channel grayscale image')

# For B channel
ax4 = fig.add_subplot(244)
ax4.imshow(B, cmap='gray')
ax4.set_title('B channel grayscale image')

# c
# histogram for R channel
fig1 = plt.figure(figsize=(15,5))
ax1 = fig1.add_subplot(131)
ax1.hist(R.ravel(), 256, [0, 256], facecolor = 'r')
ax1.set_title('R histogram')

# histogram for G channel
ax2 = fig1.add_subplot(132)
ax2.hist(G.ravel(), 256, [0, 256], facecolor = 'g')
ax2.set_title('G histogram')

# histogram for B channel
ax3 = fig1.add_subplot(133)
ax3.hist(B.ravel(), 256, [0, 256])
ax3.set_title('B histogram')

# d
# apply hist equalisation to three channels
equalised_R = cv2.equalizeHist(R)
equalised_G = cv2.equalizeHist(G)
equalised_B = cv2.equalizeHist(B)

# combine new channels
equalised_face03 = face03.copy()
equalised_face03[:, :, 0] = equalised_R
equalised_face03[:, :, 1] = equalised_G
equalised_face03[:, :, 2] = equalised_B

# plot
# new face01 with histogram equalisation
ax5 = fig.add_subplot(245)
ax5.imshow(equalised_face03)
ax5.set_title('face03 with histogram equalisation')

# new R with histogram equalisation
ax6 = fig.add_subplot(246)
ax6.imshow(equalised_R, cmap='gray')
ax6.set_title('R channel grayscale image with histogram equalisation', fontsize = 7)

# new G with histogram equalisation
ax7 = fig.add_subplot(247)
ax7.imshow(equalised_G, cmap='gray')
ax7.set_title('G channel grayscale image with histogram equalisation', fontsize = 7)

# new B with histogram equalisation
ax8 = fig.add_subplot(248)
ax8.imshow(equalised_B, cmap='gray')
ax8.set_title('B channel grayscale image with histogram equalisation', fontsize = 7)

plt.show()
cv2.waitKey(0)