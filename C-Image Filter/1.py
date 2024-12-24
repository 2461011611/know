import cv2
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread('1.jpg')

# 应用不同滤波器
# 均值滤波
average_blur = cv2.blur(img, (9, 9))
# 中值滤波
median_blur = cv2.medianBlur(img, 9)
# 高斯滤波
gaussian_blur = cv2.GaussianBlur(img, (9, 9), 0)

# 将 BGR 图像转换为 RGB 图像用于 matplotlib 显示
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
average_rgb = cv2.cvtColor(average_blur, cv2.COLOR_BGR2RGB)
median_rgb = cv2.cvtColor(median_blur, cv2.COLOR_BGR2RGB)
gaussian_rgb = cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB)

# 使用 matplotlib 显示多种滤波效果
plt.figure(figsize=(12, 8))

# 原始图像
plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title('Original')
plt.axis('off')

# 均值滤波
plt.subplot(2, 2, 2)
plt.imshow(average_rgb)
plt.title('Average Blur')
plt.axis('off')

# 中值滤波
plt.subplot(2, 2, 3)
plt.imshow(median_rgb)
plt.title('Median Blur')
plt.axis('off')

# 高斯滤波
plt.subplot(2, 2, 4)
plt.imshow(gaussian_rgb)
plt.title('Gaussian Blur')
plt.axis('off')

# 显示图像
plt.tight_layout()
plt.show()
