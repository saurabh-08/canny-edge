# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
# from google.colab import drive
# drive.mount('/content/drive')

# Loading grayscale image
# to run the code, upload an image and update the img name below
img = cv2.imread('/content/232038.jpg', 0)

sigma = 1.0  # variation performed to find the effects

# creating Gaussian fn
def gaussian_fn(x, sigma):
    two_pi_sig_sq = 2 * np.pi * sigma ** 2
    exp = - ((x ** 2) / (2 * sigma ** 2))
    return 1.0 / two_pi_sig_sq * np.exp(exp)

# Derivative of the Gaussian
def dgaussian(x, sigma):
    coeff = -x / (sigma ** 3)
    exp = np.exp(- (x ** 2) / (2 * sigma ** 2))
    normalization = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    return (exp * coeff * normalization)


# Gaussian Mask creation

x_values = np.arange(-3 * sigma, 3 * sigma + 1, 1)
g = gaussian_fn(x_values, sigma)

x_range = np.arange(-3 * sigma, 3 * sigma + 1, 1)
g_x = dgaussian(x_range, sigma)


#creating the convolution function
def convolution(kernel, image):
  height = image.shape[0]
  width = image.shape[1]
  convolution_result = np.zeros((height, width))

  # Along x
  for i in range(2, height-2):
    for j in range(2, width-2):
      for k in range(0, 5): #filter size is fixed = 5
        temp1 += kernel[k] * image[i-2+k][j]
      convolution_result[i][j] = temp1

  # Along y
  for i in range(2, height-2):
    for j in range(2, width-2):
      for k in range(0, 5): #filter size is fixed = 5
        temp2 += kernel[k] * image[i][j-2+k]
      convolution_result[i][j] = temp2

# Convolution with derivative of Gaussian

# for obtaining Ix
Ix = ndimage.convolve(img.astype(float), g_x.reshape(1,-1))

# for obtaining Iy
Iy = ndimage.convolve(img.astype(float), g_x.reshape(-1,1))

# Convolution with Ix and Iy
# for obtaining Ix prime
Ix_prime = ndimage.convolve(Ix, g.reshape(1,-1))

# for obtaining Iy prime
Iy_prime = ndimage.convolve(Iy, g.reshape(-1,1))

# Expresion for computing the magnitude
magnitude = np.sqrt(np.square(Ix_prime) + np.square(Iy_prime))

# Performing Non max suppression to further enhance
theta = np.arctan2(Iy_prime, Ix_prime)
theta_quant = (np.round(theta * (4.0 / np.pi)) + 4) % 4
nms = np.zeros_like(magnitude)
for i in range(1, magnitude.shape[0] - 1):
    for j in range(1, magnitude.shape[1] - 1):
        if(theta_quant[i,j] == 0):
            if(magnitude[i,j] >= max(magnitude[i, j-1], magnitude[i, j+1])):
                nms[i,j] = magnitude[i,j]
            else:
                nms[i,j] = 0

        elif theta_quant[i,j] == 1:
            if(magnitude[i,j] >= max(magnitude[i-1, j+1], magnitude[i+1, j-1])):
                nms[i,j] = magnitude[i,j]
            else:
                nms[i,j] = 0

        elif theta_quant[i,j] == 2:
            if(magnitude[i,j] >= max(magnitude[i-1, j], magnitude[i+1, j])):
                nms[i,j] = magnitude[i,j]
            else:
                nms[i,j] = 0

        else:
            if(magnitude[i,j] >= max(magnitude[i-1, j-1], magnitude[i+1, j+1])):
                nms[i,j] = magnitude[i,j]
            else:
                nms[i,j] = 0


# Hysteresis Thresholding performed as the final step...
upper_edge_threshold = np.percentile(nms, 90)
lower_edge_threshold = upper_edge_threshold / 3.0

#creating initial edge map to store vlaues
edge_map = np.zeros_like(nms)

#strong and weak edge processing
strong_i, strong_j = np.where(nms >= upper_edge_threshold)
zeros_i, zeros_j = np.where(nms < lower_edge_threshold)
weak_i, weak_j = np.where((nms <= upper_edge_threshold) & (nms >= lower_edge_threshold))

#setting values
edge_map[strong_i, strong_j] = 255
edge_map[weak_i, weak_j] = 50
M, N = edge_map.shape

#iteration on the img
for i in range(1, M-1):
    for j in range(1, N-1):
        if edge_map[i,j] == 50:
            if 255 in [edge_map[i+1, j-1],edge_map[i+1, j],edge_map[i+1, j+1],edge_map[i, j-1],edge_map[i, j+1],edge_map[i-1, j-1],edge_map[i-1, j],edge_map[i-1, j+1]]:
                edge_map[i, j] = 255
            else:
                edge_map[i, j] = 0

# Output Images plotting in series
plt.figure(figsize=(16,12))

plt.subplot(2, 4, 1),
plt.imshow(Ix, cmap="gray"),
plt.title("Convolution X w/ Gaussian")

plt.subplot(2, 4, 2),
plt.imshow(Iy, cmap="gray"),
plt.title("Convolution Y w/ Gaussian")

plt.subplot(2, 4, 3),
plt.imshow(Ix_prime, cmap="gray"),
plt.title("Convolution X w/ dGaussian")

plt.subplot(2, 4, 4),
plt.imshow(Iy_prime, cmap="gray"),
plt.title("Convolution Y w/ dGaussian")

plt.subplot(2, 4, 5),
plt.imshow(magnitude, cmap="gray"),
plt.title("Magnitude")

plt.subplot(2, 4, 6),
plt.imshow(nms, cmap="gray"),
plt.title("Non-max suppression")

plt.subplot(2, 4, 7),
plt.imshow(edge_map, cmap="gray"),
plt.title("Hysteresis Thresholding")

plt.tight_layout()
plt.show()

#########
