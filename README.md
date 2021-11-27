# DeFold
my lesson work. using delaunay net to fold pics, and accerates it by using c.
---
deNet.py 表示使用delaunay triangle分割图像的代码

main.py 表示主要程序，会在执行中调用deNet文件，最后生成预览图和压缩格式图片

transform.py 表示从压缩格式转化为标准24位图格式

gulugulu.c包含关键算法的c语言实现版本

gulugulu.dll为由.c文件编译为dll文件，动态链接库



---

transform文件夹里存放的为压缩格式以及对一个*狗*的图片的不同程度的处理

image文件夹中存放的是一些实例，可以对比解释算法可行性
