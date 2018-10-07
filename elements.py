# 任务：需要先设置，完成后需要做清理工作（比如：释放资源）。
# 文件处理，首先需要获取一个文件句柄，从文件中读取数据，然后关闭文件句柄。
with open("/tmp/foo.txt") as file:
    data = file.read()

# numpy（Numerical Python）提供了对多维数组对象的支持
# ndarray，具有矢量运算能力，快速、节省空间。
# numpy支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。