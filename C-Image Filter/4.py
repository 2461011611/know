import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0, std=30):
    """添加高斯噪声"""
    gauss = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), gauss)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_salt_and_pepper_noise(image, amount=0.02):
    """添加椒盐噪声"""
    noisy_image = image.copy()
    num_salt = np.ceil(amount * image.size * 0.5).astype(int)
    num_pepper = np.ceil(amount * image.size * 0.5).astype(int)

    # 添加盐噪声
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1]] = 255

    # 添加椒噪声
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1]] = 0

    return noisy_image

def add_uniform_noise(image, low=-30, high=30):
    """添加均匀噪声"""
    noise = np.random.uniform(low, high, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), noise)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def apply_filters(noisy_image):
    """应用滤波器"""
    mean_filtered = cv2.blur(noisy_image, (5, 5))
    median_filtered = cv2.medianBlur(noisy_image, 5)
    gaussian_filtered = cv2.GaussianBlur(noisy_image, (5, 5), 1)
    return mean_filtered, median_filtered, gaussian_filtered

# 读取图像
image = cv2.imread('4.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 添加噪声
gaussian_noisy = add_gaussian_noise(image)
salt_and_pepper_noisy = add_salt_and_pepper_noise(image)
uniform_noisy = add_uniform_noise(image)

# 滤波处理
gaussian_mean, gaussian_median, gaussian_gaussian = apply_filters(gaussian_noisy)
sp_mean, sp_median, sp_gaussian = apply_filters(salt_and_pepper_noisy)
uniform_mean, uniform_median, uniform_gaussian = apply_filters(uniform_noisy)

# 可视化
plt.figure(figsize=(18, 12))

# 原始图像
plt.subplot(4, 4, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

# 高斯噪声及滤波
plt.subplot(4, 4, 2)
plt.imshow(gaussian_noisy)
plt.title("Gaussian Noise")
plt.axis('off')
plt.subplot(4, 4, 3)
plt.imshow(gaussian_mean)
plt.title("Mean Filter (Gaussian)")
plt.axis('off')
plt.subplot(4, 4, 4)
plt.imshow(gaussian_median)
plt.title("Median Filter (Gaussian)")
plt.axis('off')
plt.subplot(4, 4, 5)
plt.imshow(gaussian_gaussian)
plt.title("Gaussian Filter (Gaussian)")
plt.axis('off')

# 椒盐噪声及滤波
plt.subplot(4, 4, 6)
plt.imshow(salt_and_pepper_noisy)
plt.title("Salt & Pepper Noise")
plt.axis('off')
plt.subplot(4, 4, 7)
plt.imshow(sp_mean)
plt.title("Mean Filter (Salt & Pepper)")
plt.axis('off')
plt.subplot(4, 4, 8)
plt.imshow(sp_median)
plt.title("Median Filter (Salt & Pepper)")
plt.axis('off')
plt.subplot(4, 4, 9)
plt.imshow(sp_gaussian)
plt.title("Gaussian Filter (Salt & Pepper)")
plt.axis('off')

# 均匀噪声及滤波
plt.subplot(4, 4, 10)
plt.imshow(uniform_noisy)
plt.title("Uniform Noise")
plt.axis('off')
plt.subplot(4, 4, 11)
plt.imshow(uniform_mean)
plt.title("Mean Filter (Uniform)")
plt.axis('off')
plt.subplot(4, 4, 12)
plt.imshow(uniform_median)
plt.title("Median Filter (Uniform)")
plt.axis('off')
plt.subplot(4, 4, 13)
plt.imshow(uniform_gaussian)
plt.title("Gaussian Filter (Uniform)")
plt.axis('off')

plt.tight_layout()
plt.show()
