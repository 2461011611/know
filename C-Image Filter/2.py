import cv2
import matplotlib.pyplot as plt

# 读取原始图像
img = cv2.imread('1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式以适配 matplotlib 显示

# 应用不同滤波器
# 均值滤波
average_blur = cv2.blur(img, (15, 15))
# 中值滤波
median_blur = cv2.medianBlur(img, 15)
# 高斯滤波
gaussian_blur = cv2.GaussianBlur(img, (15, 15), 5)

# 使用 matplotlib 显示多种滤波效果
plt.figure(figsize=(12, 8))

# 原始图像
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

# 均值滤波
plt.subplot(2, 2, 2)
plt.imshow(average_blur)
plt.title('Average Blur')
plt.axis('off')

# 中值滤波
plt.subplot(2, 2, 3)
plt.imshow(median_blur)
plt.title('Median Blur')
plt.axis('off')

# 高斯滤波
plt.subplot(2, 2, 4)
plt.imshow(gaussian_blur)
plt.title('Gaussian Blur')
plt.axis('off')

# 显示图像
plt.tight_layout()
plt.show()
