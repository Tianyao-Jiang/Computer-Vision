# import some packages we need

import numpy as np
import cv2
import matplotlib.pyplot as plt

# task 2

# 2.1
# Load the image
img = cv2.imread('Lenna.png')
# Change it to the grey version using function
grey_I = img[:, :, 2] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 0] * 0.114
grey_I = np.uint8(grey_I)
print(grey_I)

# Use bit_wise to flip and get its negative image
negative_grey_I = cv2.bitwise_not(grey_I)
print(negative_grey_I)

# Create a figure
fig = plt.figure(figsize=(14, 10))
# Show them side by side
ax1 = fig.add_subplot(231)
ax1.imshow(grey_I, cmap='gray')
ax1.set_title('Grey Lenna')

ax2 = fig.add_subplot(232)
ax2.imshow(negative_grey_I, cmap='gray')
ax2.set_title('Negative grey Lenna')

# 2.2
# Flip vertically
flipped_I = cv2.flip(grey_I, 0)
ax3 = fig.add_subplot(233)
ax3.imshow(flipped_I, cmap='gray')
ax3.set_title('Grey Lenna flipped vertically')

# 2.3
# load a colour image
lenna = cv2.imread('Lenna.png')

# swap red and blue channels
# lenna[:, :, [0, 2]] = lenna[:, :, [2, 0]]

# Plot
ax4 = fig.add_subplot(234)
ax4.imshow(lenna)
ax4.set_title('Colourful Lenna with R and B channels swapped', fontsize = 10)

# 2.4
# Average the input image with its vertically flipped image
# vertically flip colorful Lenna
flipped_lenna = cv2.flip(lenna, 0)
# Change the type
lenna = lenna.astype(np.float32)
flipped_lenna = flipped_lenna.astype(np.float32)

# swap red and blue channels
lenna[:, :, [0, 2]] = lenna[:, :, [2, 0]]
# Mix these two images
img_mix = (lenna + flipped_lenna) / 2
# Change the mixed one to uint8
img_mix= img_mix.astype(np.uint8)
# Plot
ax5 = fig.add_subplot(235)
ax5.imshow(img_mix)
ax5.set_title('Average the input with its vertically flipped one', fontsize = 10)

# 2.5
# Add a random value between [0,255] to every pixel in the grayscale image, then clip the new image to
# have a minimum value of 0 and a maximum value of 255.

# Create a matrix of random noise range of 0 to 255
random_matrix = np.random.randint(0, 255, size= grey_I.shape)
# Adding the noise
random_I = grey_I + random_matrix

# Clipping
random_I = np.clip(random_I, 0, 255)

# change the type back to uint8
random_I = random_I.astype(np.uint8)

ax6 = fig.add_subplot(236)
ax6.imshow(random_I, cmap='gray')
ax6.set_title('Grey Lenna with random noise added', fontsize = 10)

plt.show()
cv2.waitKey(0)
