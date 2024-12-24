import cv2
import matplotlib.pyplot as plt

# 读取原始图像
img = cv2.imread('1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式以适配 matplotlib 显示

# 应用不同滤波器
# 大核均值滤波
average_blur = cv2.blur(img, (31, 31))
# 大核中值滤波
median_blur = cv2.medianBlur(img, 31)
# 大核高斯滤波
gaussian_blur = cv2.GaussianBlur(img, (31, 31), 15)

# 计算滤波前后的差值图像
diff_average = cv2.absdiff(img, average_blur)
diff_median = cv2.absdiff(img, median_blur)
diff_gaussian = cv2.absdiff(img, gaussian_blur)

# 使用 matplotlib 显示滤波效果和差值
plt.figure(figsize=(16, 12))

# 原始图像
plt.subplot(3, 3, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

# 均值滤波
plt.subplot(3, 3, 2)
plt.imshow(average_blur)
plt.title('Average Blur')
plt.axis('off')

# 均值滤波差值
plt.subplot(3, 3, 3)
plt.imshow(diff_average)
plt.title('Average Diff')
plt.axis('off')

# 中值滤波
plt.subplot(3, 3, 4)
plt.imshow(median_blur)
plt.title('Median Blur')
plt.axis('off')

# 中值滤波差值
plt.subplot(3, 3, 5)
plt.imshow(diff_median)
plt.title('Median Diff')
plt.axis('off')

# 高斯滤波
plt.subplot(3, 3, 6)
plt.imshow(gaussian_blur)
plt.title('Gaussian Blur')
plt.axis('off')

# 高斯滤波差值
plt.subplot(3, 3, 7)
plt.imshow(diff_gaussian)
plt.title('Gaussian Diff')
plt.axis('off')

# 调整布局
plt.tight_layout()
plt.show()
