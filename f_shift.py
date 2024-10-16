import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像，并将其转换为灰度图
image = cv2.imread('322618.jpg', cv2.IMREAD_GRAYSCALE)

# 进行傅里叶变换
f_transform = np.fft.fft2(image)  # 2D 傅里叶变换
f_shift = np.fft.fftshift(f_transform)  # 将低频分量移动到中心
magnitude_spectrum = 20 * np.log(np.abs(f_shift))  # 计算频谱图

# 显示原始图像和傅里叶频谱图
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Fourier Transform (Magnitude Spectrum)')
plt.axis('off')

plt.show()
