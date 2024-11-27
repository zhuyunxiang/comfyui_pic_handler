from .crop_pic import *

NODE_CLASS_MAPPINGS = {
  'ZYX图像裁剪': CropPic,
  'ZYX图像梯形变换': TrapezoidalTransform,
  'ZYX图像上方指定高度右转': SkewImageTopRight,
  'ZYX图像上方指定高度左转': SkewImageTopLeft,
  'ZYX图像上方和下方指定高度左转': SkewImageTopBottomLeft,
  'ZYX图像上方和下方指定高度右转': SkewImageTopBottomRight,
  'ZYX图像旋转角度（带扩图）': RotateImage,
  'ZYX图像resize（图像会伸缩）': ResizeImage,
  'ZYX图像resize（图像保持纵横比）': ResizeWithRatioImage,
  'ZYX图像向四周扩图': ExpandImage,
  'ZYX清除图像四周的空白并调整图像大小': ClearImageBorder,
  'ZYX获取图像四周的空白尺寸': GetImageBlankSize,
  'ZYX计算非空白图像尺寸': GetImageSizeWithoutBorder,
  'ZYX根据示例图调整尺寸': AdjustImageSize,
  'ZYX图像局部压缩': SplitCompressTransform,
  'ZYX图像局部压缩(去锯齿)': SplitCompressTransformNo,
  'ZYX图像扩展': ResizeImageOffset,
}

__all__ = ['NODE_CLASS_MAPPINGS']


