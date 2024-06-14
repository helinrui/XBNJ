# 1、Easy and Precise Segmentation-Guided Diffusion Models

## 配置环境、数据准备、训练、测试阅读文件夹里面的readme文件

##  注
1、在原有代码基础上解决了针对RGB三通道图像加颜色mask引导时的BUG，其他未做改动
2、difussion22文件夹包含细胞内镜第二类图片（原本39张和风格迁移1208），直接放到segmentation-guided-diffusion-main/路径下
3、difussion22_mask文件夹是颜色掩码，与difussion22中图片一一对应，直接放到segmentation-guided-diffusion-main/路径下


# 2、Rethinking Transfer Learning for Medical Image Classification(TTL)

##  配置环境、数据准备、训练、测试阅读文件夹里面的readme文件

## 注：在原有代码基础上加了
1、自己写了一个XBNJ.py(针对细胞内镜数据加载器)，在TTL/utils/路径下
2、增加了--resume参数(在训练中断时重新加载训练模型)
3、自己写了一个test.py(测试已经训练好的模型)，在TTL/路径下
4、数据集图片和对应的label放到TTL/data/路径下

# 3、StyleID

##  配置环境、数据准备、训练、测试阅读文件夹里面的readme文件

## 3）注：没有改动源代码，阅读文件夹里面的readme文件即可（运行此代码需要显存大于20M）
1、把stable-diffusion-v1文件放到StyleID/models/ldm/路径下
2、把taming-transformers、openai、CLIP三个文件夹放到StyleID/路径下
3、分别进入taming-transformers、CLIP两个文件夹文件目录下运行pip install -e .命令即可运行


