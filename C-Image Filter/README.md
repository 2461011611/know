1.平滑滤波处理是一种常用的图像处理技术，主要用于去除图像中的噪声和相机失真。常见的平滑滤波方法包括均值滤波、中值滤波、高斯滤波和双边滤波。

2.均值滤波是一种简单的平滑方法，通过对图像中一定邻域内的像素灰度值求平均值，将平均的结果作为中心像素的灰度保存在结果图中

3.中值滤波是一种非线性滤波方法，适合去除椒盐噪声。其主要思想是对中心像素矩形邻域取中值来替代中心像素。

4.高斯滤波是一种常用的滤波方法，适合处理高斯噪声。其主要思想是使用高斯核与输入图像中的每个点作卷积运算，然后对卷积结果进行求和，从而得到输出图像。

5.双边滤波是一种可以保持图像边缘的平滑方法。其权值由两部分组成：一部分与高斯平滑的权值相同，另一部分基于和中心像素点的灰度值的差异。


# 1.py  

# 输出结果：将原始图像和三种滤波效果（均值滤波、中值滤波和高斯滤波）放在一个界面中对比显示

![image](https://github.com/user-attachments/assets/efd80558-7992-43a6-bdaa-5c1f5e3c9dce)

# 2.py

##图像加入随机噪声后的效果，以及通过三种滤波器（均值滤波、中值滤波和高斯滤波）处理噪声后的图像。

![image](https://github.com/user-attachments/assets/14a66707-3fd8-4459-a831-8f33823781e1)

# 3.py

##应用了不同的滤波器（均值、中值和高斯滤波）到图像上，并计算了原始图像与滤波后的图像之间的差异，然后使用 matplotlib 可视化了原始图像、滤波图像和差异图像

![image](https://github.com/user-attachments/assets/c093c0fa-1887-46fd-83b4-5b387677e95f)

# 4.py
##实现了三种常见噪声（高斯噪声、椒盐噪声和均匀噪声）的添加，并应用了三种滤波器（均值滤波、中值滤波和高斯滤波）来处理这些噪声。

![image](https://github.com/user-attachments/assets/562acfe8-d0b7-496b-bbe9-1ed85690f09e)

