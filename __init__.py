from .crop_pic import *

NODE_CLASS_MAPPINGS = {
  'ZYX图像裁剪': CropPic,
  'ZYX图像梯形变换': TrapezoidalTransform,
  'ZYX图像上方指定高度右转': SkewImageTopRight,
  'ZYX图像上方指定高度左转': SkewImageTopLeft,
  'ZYX图像上方和下方指定高度左转': SkewImageTopBottomLeft,
  'ZYX图像上方和下方指定高度右转': SkewImageTopBottomRight,
}

__all__ = ['NODE_CLASS_MAPPINGS']


