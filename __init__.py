from .crop_pic import *

NODE_CLASS_MAPPINGS = {
  'ZYX图像裁剪': CropPic,
  'ZYX图像梯形变换': TrapezoidalTransform,
  'ZYX图像上方指定高度右转': SkewImageTopRight,
}

__all__ = ['NODE_CLASS_MAPPINGS']


