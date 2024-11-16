from PIL import Image
import torch
from collections.abc import Callable
import numpy as np
import numpy.typing as npt
import math

def to_numpy(image: torch.Tensor) -> npt.NDArray[np.uint8]:
    np_array = np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
    return np_array

def handle_batch(
    tensor: torch.Tensor,
    func: Callable[[torch.Tensor], Image.Image | npt.NDArray[np.uint8]],
) -> list[Image.Image] | list[npt.NDArray[np.uint8]]:
    """Handles batch processing for a given tensor and conversion function."""
    return [func(tensor[i]) for i in range(tensor.shape[0])]

def tensor2pil(tensor: torch.Tensor) -> list[Image.Image]:
    """Converts a batch of tensors to a list of PIL Images."""

    def single_tensor2pil(t: torch.Tensor) -> Image.Image:
        np_array = to_numpy(t)
        if np_array.ndim == 2:  # (H, W) for masks
            return Image.fromarray(np_array, mode="L")
        elif np_array.ndim == 3:  # (H, W, C) for RGB/RGBA
            if np_array.shape[2] == 3:
                return Image.fromarray(np_array, mode="RGB")
            elif np_array.shape[2] == 4:
                return Image.fromarray(np_array, mode="RGBA")
        raise ValueError(f"Invalid tensor shape: {t.shape}")

    return handle_batch(tensor, single_tensor2pil)

def pil2tensor(images: Image.Image | list[Image.Image]) -> torch.Tensor:
    """Converts a PIL Image or a list of PIL Images to a tensor."""

    def single_pil2tensor(image: Image.Image) -> torch.Tensor:
        np_image = np.array(image).astype(np.float32) / 255.0
        if np_image.ndim == 2:  # Grayscale
            return torch.from_numpy(np_image).unsqueeze(0)  # (1, H, W)
        else:  # RGB or RGBA
            return torch.from_numpy(np_image).unsqueeze(0)  # (1, H, W, C)

    if isinstance(images, Image.Image):
        return single_pil2tensor(images)
    else:
        return torch.cat([single_pil2tensor(img) for img in images], dim=0)

# 图片裁剪
class CropPic:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "display": 'number'
                }),
                "isTop": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 1,
                    "step": 1,
                    "display": 'number'
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    CATEGORY = "img_process"
    FUNCTION = "crop_img"

    def crop_img(self, image, crop_height, isTop):
        # 加载图像
        img = tensor2pil(image)[0].convert("RGBA")
        width, height = img.size

        # 确保裁剪高度不超过图像高度
        if crop_height > height:
            raise ValueError("裁剪高度不能超过图像高度。")

        if isTop == 1:
            # 从顶部裁剪
            cropped_img = img.crop((0, crop_height, width, height))
        else:
            # 从底部裁剪
            cropped_img = img.crop((0, 0, width, height - crop_height))


        return (pil2tensor(cropped_img), )  # 返回裁剪后的图像作为张量

# 梯形变换节点
class TrapezoidalTransform(object):
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "bottom_half_height_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "display": 'float'
                }),
                "top_width_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "display": 'float'
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    CATEGORY = "img_process"
    FUNCTION = "trapezoidal_transform"

    # 将图像转换为梯形
    def trapezoidal_transform(self, image, bottom_half_height_ratio=0.5, top_width_ratio=0.6):
        # 加载图像
        # 加载图像
        img = tensor2pil(image)[0].convert("RGBA")
        width, height = img.size

        # 计算下半部分的高度
        bottom_half_height = int(height * bottom_half_height_ratio)
        top_half_height = height - bottom_half_height

        # 创建新的图像，初始化为透明
        new_img = Image.new('RGBA', img.size)

        # 定义梯形的顶部宽度
        top_width = int(width * top_width_ratio)

        # 处理上半部分（梯形部分）
        for y in range(top_half_height):
            # 计算当前行的宽度
            current_width = int(top_width + (width - top_width) * (y / top_half_height))
            left = (width - current_width) // 2
            right = left + current_width
            
            # 逐个像素进行插值
            for x in range(left, right):
                # 计算原图中对应的像素位置
                source_x = int((x - left) * (width / current_width))  # 线性映射
                if 0 <= source_x < width and 0 <= y < height:
                    new_img.putpixel((x, y), img.getpixel((source_x, y)))

        # 处理下半部分（保持不变）
        new_img.paste(img.crop((0, top_half_height, width, height)), (0, top_half_height))

        # 保存处理后的图像
        return (pil2tensor(new_img.convert("RGB")), )

# 上半部分往右偏移节点
class SkewImageTopRight(object):
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "skew_angle": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 90,
                    "step": 1,
                    "display": 'number'
                }),
                "bottom_half_height_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "display": 'float'
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    CATEGORY = "img_process"
    FUNCTION = "skew_image_top_right"

    # 上半部分往右偏移
    def skew_image_top_right(self, image, skew_angle=30, bottom_half_height_ratio=0.5):
        # 加载图像
        img = tensor2pil(image)[0].convert("RGBA")
        width, height = img.size

        # 计算下半部分的高度
        bottom_half_height = int(height * bottom_half_height_ratio)
        top_half_height = height - bottom_half_height

        # 将倾斜角度转换为弧度
        angle_rad = math.radians(skew_angle)

        # 计算新宽度
        total_offset = int(top_half_height * math.tan(angle_rad))
        new_width = width + total_offset

        # 创建新的图像，初始化为透明
        new_img = Image.new('RGBA', (new_width, height))

        # 处理上半部分（平行四边形部分）
        for y in range(top_half_height):
            # 计算当前行的偏移量，向右倾斜
            offset = int((top_half_height - y) * math.tan(angle_rad))

            for x in range(width):
                # 计算新的x坐标，保持底边不变
                new_x = x + offset
                
                # 确保坐标在有效范围内
                if 0 <= new_x < new_width and 0 <= y < top_half_height:
                    new_img.putpixel((new_x, y), img.getpixel((x, y)))

        # 处理下半部分（保持不变）
        new_img.paste(img.crop((0, top_half_height, width, height)), (0, top_half_height))

        return (pil2tensor(new_img.convert("RGB")), )

# 上半部分往左偏移节点
class SkewImageTopLeft(object):
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "skew_angle": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 90,
                    "step": 1,
                    "display": 'number'
                }),
                "bottom_half_height_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "display": 'float'
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    CATEGORY = "img_process"
    FUNCTION = "skew_image_top_left"
    
    # 上半部分往左偏移
    def skew_image_top_left(self, image, skew_angle=30, bottom_half_height_ratio=0.5):
        img = tensor2pil(image)[0].convert("RGBA")
        width, height = img.size

        # 计算下半部分的高度
        bottom_half_height = int(height * bottom_half_height_ratio)
        top_half_height = height - bottom_half_height

        # 将倾斜角度转换为弧度
        angle_rad = math.radians(skew_angle)

        # 计算新宽度，向左倾斜时需要向右扩展宽度
        total_offset = int(top_half_height * math.tan(angle_rad))
        new_width = width + total_offset

        # 创建新的图像，初始化为透明
        new_img = Image.new('RGBA', (new_width, height))

        # 处理上半部分（平行四边形部分）
        for y in range(top_half_height):
            # 计算当前行的偏移量，向左倾斜
            offset = int((top_half_height - y) * math.tan(angle_rad))

            for x in range(width):
                # 计算新的x坐标，保持底边与下半部分重合，同时整体向右移动
                new_x = x + total_offset - offset
                
                # 确保坐标在有效范围内
                if 0 <= new_x < new_width and 0 <= y < top_half_height:
                    new_img.putpixel((new_x, y), img.getpixel((x, y)))

        # 处理下半部分（保持不变）
        new_img.paste(img.crop((0, top_half_height, width, height)), (total_offset, top_half_height))

        # 保存处理后的图像
        return (pil2tensor(new_img.convert("RGB")), )
