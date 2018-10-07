import numpy as np
import matplotlib.pyplot as plt # 用于图像显示
from skimage import filters, io
import matplotlib.cm as cm
import scipy.signal as signal

filepath = r'..\images\Liver.BMP' # input("请输入文件路径：")
image = io.imread(filepath, as_grey=True)

# Image对象转化成图像矩阵
image_array = np.array(image)
# # 绘制直方图并观察
plt.hist(image_array.flatten(), 256)
plt.show()

def histeq(image_array, image_bins=256):
    # 将图像矩阵转化成直方图数据，返回元组(频数，直方图区间坐标)
    image_array2, bins = np.histogram(image_array.flatten(), image_bins)

    # 计算直方图的累积函数
    cdf = image_array2.cumsum()

    # 将累积函数转化到区间[0,255]
    cdf = (255.0 / cdf[-1]) * cdf

    # 原图像矩阵利用累积函数进行转化，插值过程
    image2_array = np.interp(image_array.flatten(), bins[:-1], cdf)

    # 返回均衡化后的图像矩阵和累积函数
    return image2_array.reshape(image_array.shape), cdf

# 利用定义的直方图均衡化函数对图像进行均衡化处理
image_new_array = histeq(image_array)[0]

# 绘制均衡化后的直方图
plt.hist(image_new_array[0].flatten(), 256)

# # 绘制均衡化后的图像
image_new = Image.fromarray(image_new_array[0])
plt.imshow(image_new, cmap=cm.gray)
plt.axis("off")
plt.show()

thresh = filters.threshold_isodata(image_new_array,7,'median')   #返回一个阈值
dst =(image_new_array >= thresh)*1.0   #根据阈值进行分割

# plt.figure('thresh',figsize=(8,8))
# plt.subplot(121)
# plt.title('original image')
# plt.axis("off")
# plt.imshow(image_gray,cmap=cm.gray)
# plt.subplot(122)
# plt.title('binary image')
# plt.axis("off")
# plt.imshow(dst,cmap=cm.gray)
# plt.show()


# 生成高斯算子的函数
def func(x,y,sigma=1):
    return 100*(1/(2*np.pi*sigma))*np.exp(-((x-2)**2+(y-2)**2)/(2.0*sigma**2))

# 生成标准差为2的5*5高斯算子
suanzi = np.fromfunction(func,(5,5),sigma=2)

# 图像与高斯算子进行卷积
image2 = signal.convolve2d(dst,suanzi,mode="same")

# 结果转化到0-255
image2 = (image2/float(image2.max()))*255

# 显示图像
plt.subplot(2,1,1)
plt.imshow(image_array,cmap=cm.gray)
plt.axis("off")
plt.subplot(2,1,2)
plt.imshow(image2,cmap=cm.gray)
plt.axis("off")
plt.show()