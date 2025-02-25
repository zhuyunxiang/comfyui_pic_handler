from PIL import Image, ImageOps, ImageFilter, ImageDraw
import torch
from collections.abc import Callable
import numpy as np
import numpy.typing as npt
import math
import cv2

def pil_to_cv2(pil_img):
    # 将 PIL 图像转换为 NumPy 数组
    open_cv_img = np.array(pil_img)

    # 检查是否有 alpha 通道
    if open_cv_img.shape[2] == 4:  # RGBA 图像
        # 转换为 BGRA
        open_cv_img = cv2.cvtColor(open_cv_img, cv2.COLOR_RGBA2BGRA)
    else:
        # RGB 转换为 BGR
        open_cv_img = cv2.cvtColor(open_cv_img, cv2.COLOR_RGB2BGR)

    return open_cv_img

def cv2_to_pil(cv2_img):
    # 检查是否有 alpha 通道
    if cv2_img.shape[2] == 4:  # BGRA 图像
        # 转换为 RGBA
        pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGRA2RGBA))
    else:
        # BGR 转换为 RGB
        pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

    return pil_img

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
                "isRGB": ("INT", {
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

    def crop_img(self, image, crop_height, isTop, isRGB):
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

        if isRGB == 1:
            return (pil2tensor(cropped_img.convert("RGB")), )

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
                    "default": 0.6,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "display": 'float'
                }),
                "isRGB": ("INT", {
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
    FUNCTION = "trapezoidal_transform"

    # 将图像转换为梯形
    def trapezoidal_transform(self, image, bottom_half_height_ratio=0.5, top_width_ratio=0.6, isRGB=0):
        # 将图像转换为 RGBA 格式
        pil_img = tensor2pil(image)[0].convert("RGBA")
        img = pil_to_cv2(pil_img)

        original_height, original_width = img.shape[:2]
        print(f"Original Image Size: Width={original_width}, Height={original_height}")

        # Calculate top height and target width
        bottom_half_height = int(original_height * bottom_half_height_ratio)
        top_height = original_height - bottom_half_height
        target_width = int(original_width * top_width_ratio)
        print(f"Bottom Half Height: {bottom_half_height}, Top Height: {top_height}, Target Width for Top: {target_width}")

        # Define source and target points
        src_points = np.float32([
            [0, 0],                 # Top-left
            [original_width, 0],    # Top-right
            [0, top_height],         # Bottom-left
            [original_width, top_height] # Bottom-right
        ])

        # Set target points, top narrowed
        target_points = np.float32([
            [original_width // 2 - target_width // 2, 0],  # New top-left
            [original_width // 2 + target_width // 2, 0],  # New top-right
            [0, top_height],                             # Bottom-left
            [original_width, top_height]                    # Bottom-right
        ])

        print("Source Points:", src_points)
        print("Target Points:", target_points)

        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_points, target_points)

        # Apply perspective transform
        new_img = cv2.warpPerspective(img, matrix, (original_width, original_height))

        # Create image with transparent background, initialized as fully transparent
        transparent_background = np.zeros((original_height, original_width, 4), dtype=np.uint8)

        # Handle the bottom half of the original image
        if img.shape[2] == 4:  # RGBA image
            transparent_background[top_height:, :] = img[top_height:, :]  # Copy RGBA part
        else:  # RGB image
            bottom_half_rgba = np.concatenate((
                img[top_height:, :],
                np.full((img[top_height:].shape[0], img[top_height:].shape[1], 1), 255, dtype=np.uint8)  # Add alpha channel
            ), axis=2)
            transparent_background[top_height:, :] = bottom_half_rgba

        # Copy the transformed top half to the transparent background
        if new_img.shape[2] == 4:  # Transformed image is RGBA
            transparent_background[:top_height, :] = new_img[:top_height, :]
        else:  # Transformed image is RGB
            top_half_rgba = np.concatenate((
                new_img[:top_height, :],
                np.full((new_img[:top_height].shape[0], new_img[:top_height].shape[1], 1), 255, dtype=np.uint8)  # Add alpha channel
            ), axis=2)
            transparent_background[:top_height, :] = top_half_rgba

        # Set the areas not covered by the image to transparent
        # transparent_background[:, :, 3] = np.where(transparent_background[:, :, :3].sum(axis=2) > 0, 255, 0)

        # Check if output image is valid
        if transparent_background is None or transparent_background.size == 0 or np.count_nonzero(transparent_background) == 0:
            print("Error: The transformed image is empty or not valid.")
            return

        res_pil_img = cv2_to_pil(transparent_background)

        if isRGB == 0:
            new_img = res_pil_img.convert("RGBA")

        if isRGB == 1:
            new_img = res_pil_img.convert("RGB")

        # Save the processed image
        return (pil2tensor(new_img), )

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
                "isRGB": ("INT", {
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
    FUNCTION = "skew_image_top_right"

    # 上半部分往右偏移
    def skew_image_top_right(self, image, skew_angle=30, bottom_half_height_ratio=0.5, isRGB=1):
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

        # 使用抗锯齿方法平滑边缘
        if isRGB == 1:
            new_img = new_img.convert("RGB")

        return (pil2tensor(new_img.filter(ImageFilter.SMOOTH)), )

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
                "isRGB": ("INT", {
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
    FUNCTION = "skew_image_top_left"

    def skew_image_top_left(self, image, skew_angle=30, bottom_half_height_ratio=0.5, isRGB=1):
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
        draw = ImageDraw.Draw(new_img)
        
        for y in range(top_half_height):
            offset = int((top_half_height - y) * math.tan(angle_rad))
            
            for x in range(width):
                new_x = x + total_offset - offset
                
                if 0 <= new_x < new_width and 0 <= y < top_half_height:
                    color = img.getpixel((x, y))
                    draw.point((new_x, y), fill=color)

        # 处理下半部分（保持不变）
        new_img.paste(img.crop((0, top_half_height, width, height)), (total_offset, top_half_height))

        # 抗锯齿处理
        new_img = new_img.filter(ImageFilter.SMOOTH_MORE)

        if isRGB == 1:
            return (pil2tensor(new_img.convert("RGB")), )

        # 保存处理后的图像
        return (pil2tensor(new_img), )

# 上半部分往左偏移下半部分也偏移节点
class SkewImageTopBottomLeft(object):
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "skew_angle_top": ("INT", {
                    "default": 30,
                    "min": 0,
                    "max": 90,
                    "step": 1,
                    "display": 'number'
                }),
                "skew_angle_bottom": ("INT", {
                    "default": -15,
                    "min": -90,
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
                "isRGB": ("INT", {
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
    FUNCTION = "skew_image_top_bottom_left"

    def skew_image_top_bottom_left(self, image, skew_angle_top=30, skew_angle_bottom=-15, bottom_half_height_ratio=0.5, isRGB=1):
        # 加载图像
        img = tensor2pil(image)[0].convert("RGBA")
        
        # 获取原始宽度和高度
        width, height = img.size

        # 计算下半部分的高度
        bottom_half_height = int(height * bottom_half_height_ratio)
        top_half_height = height - bottom_half_height

        # 将倾斜角度转换为弧度
        angle_rad_top = math.radians(skew_angle_top)
        angle_rad_bottom = math.radians(skew_angle_bottom)

        # 计算新宽度
        total_offset_top = int(top_half_height * math.tan(angle_rad_top))
        total_offset_bottom = int(bottom_half_height * math.tan(angle_rad_bottom))
        new_width = width + max(total_offset_top, total_offset_bottom)

        # 创建新的图像，初始化为透明
        new_img = Image.new('RGBA', (new_width, height))

        # 处理上半部分（平行四边形部分）
        for y in range(top_half_height):
            # 计算当前行的偏移量，向左倾斜
            offset_top = int((top_half_height - y) * math.tan(angle_rad_top))
            
            for x in range(width):
                # 计算新的x坐标，保持底边与下半部分重合
                new_x = x + total_offset_top - offset_top
                
                # 确保坐标在有效范围内
                if 0 <= new_x < new_width and 0 <= y < top_half_height:
                    new_img.putpixel((new_x, y), img.getpixel((x, y)))

        # 处理下半部分（保持上边不变，整体向右移动以与上半部分对齐）
        for y in range(bottom_half_height):
            # 计算当前行的偏移量，向右倾斜
            offset_bottom = int(y * math.tan(angle_rad_bottom))
            
            for x in range(width):
                # 计算新的x坐标，整体向右移动以与上半部分对齐
                new_x = x + total_offset_top + offset_bottom  # 向右移动
                
                # 确保坐标在有效范围内
                if 0 <= new_x < new_width and top_half_height + y < height:
                    new_img.putpixel((new_x, top_half_height + y), img.getpixel((x, top_half_height + y)))

        # 对处理后的图像进行平滑处理，减少锯齿效果
        new_img = new_img.filter(ImageFilter.SMOOTH)

        if isRGB == 1:
            return (pil2tensor(new_img.convert("RGB")),)
        
        return (pil2tensor(new_img),)

# 上半部分往左偏移下半部分也偏移节点（带调整高度）
class SkewImageLeft(object):
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "skew_angle_top": ("INT", {
                    "default": 30,
                    "min": 0,
                    "max": 90,
                    "step": 1,
                    "display": 'number'
                }),
                "skew_angle_bottom": ("INT", {
                    "default": -15,
                    "min": -90,
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
                "top_half_height_ratio": ("FLOAT", {
                    "default": 1,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "display": 'float'
                }),
                "isRGB": ("INT", {
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
    FUNCTION = "skew_image_left"

    def skew_image_left(self, image, skew_angle_top=30, skew_angle_bottom=-15, bottom_half_height_ratio=0.5, top_half_height_ratio=1, isRGB=1):
        # 加载图像
        img = tensor2pil(image)[0].convert("RGBA")
        
        # 获取原始宽度和高度
        width, height = img.size

        # 计算下半部分的高度
        bottom_half_height = int(height * bottom_half_height_ratio)
        top_half_height = height - bottom_half_height

        # 调整上半部分高度
        adjusted_top_half_height = int(top_half_height * top_half_height_ratio)

        # 将倾斜角度转换为弧度
        angle_rad_top = math.radians(skew_angle_top)
        angle_rad_bottom = math.radians(skew_angle_bottom)

        # 计算新宽度
        total_offset_top = int(adjusted_top_half_height * math.tan(angle_rad_top))
        total_offset_bottom = int(bottom_half_height * math.tan(angle_rad_bottom))
        new_width = width + max(total_offset_top, total_offset_bottom)

        # 创建新的图像，初始化为透明
        new_img = Image.new('RGBA', (new_width, height))

        # 处理上半部分（平行四边形部分）
        for y in range(adjusted_top_half_height):
            # 计算当前行的偏移量，向左倾斜
            offset_top = int((adjusted_top_half_height - y) * math.tan(angle_rad_top))
            
            for x in range(width):
                # 计算新的x坐标，保持底边与下半部分重合
                new_x = x + total_offset_top - offset_top
                
                # 确保坐标在有效范围内
                if 0 <= new_x < new_width and 0 <= y < adjusted_top_half_height:
                    # 计算原图的y坐标，依据压缩或拉伸比例
                    original_y = int(y / top_half_height_ratio)
                    
                    # 确保原图的y坐标在有效范围内
                    if original_y < top_half_height:
                        new_img.putpixel((new_x, y), img.getpixel((x, original_y)))

        # 处理下半部分（保持上边不变，整体向右移动以与上半部分对齐）
        for y in range(bottom_half_height):
            # 计算当前行的偏移量，向右倾斜
            offset_bottom = int(y * math.tan(angle_rad_bottom))
            
            for x in range(width):
                # 计算新的x坐标，整体向右移动以与上半部分对齐
                new_x = x + total_offset_top + offset_bottom  # 向右移动
                
                # 确保坐标在有效范围内
                if 0 <= new_x < new_width and adjusted_top_half_height + y < height:
                    new_img.putpixel((new_x, adjusted_top_half_height + y), img.getpixel((x, top_half_height + y)))

        # 对处理后的图像进行平滑处理，减少锯齿效果
        new_img = new_img.filter(ImageFilter.SMOOTH)

        if isRGB == 1:
            return (pil2tensor(new_img.convert("RGB")),)
        
        return (pil2tensor(new_img),)

# 上半部分往右偏移下半部分也偏移节点（带调整高度）
class SkewImageRight(object):
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "skew_angle_top": ("INT", {
                    "default": 30,
                    "min": 0,
                    "max": 90,
                    "step": 1,
                    "display": 'number'
                }),
                "skew_angle_bottom": ("INT", {
                    "default": -15,
                    "min": -90,
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
                "top_half_height_ratio": ("FLOAT", {
                    "default": 1,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "display": 'float'
                }),
                "isRGB": ("INT", {
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
    FUNCTION = "skew_image_right"

    def skew_image_right(self, image, skew_angle_top=30, skew_angle_bottom=-15, bottom_half_height_ratio=0.5, top_half_height_ratio=1, isRGB=1):
        # 加载图像
        img = tensor2pil(image)[0].convert("RGBA")
        width, height = img.size

        # 计算下半部分的高度
        bottom_half_height = int(height * bottom_half_height_ratio)
        top_half_height = height - bottom_half_height

        # 调整上半部分高度
        adjusted_top_half_height = int(top_half_height * top_half_height_ratio)

        # 将倾斜角度转换为弧度
        angle_rad_top = math.radians(skew_angle_top)
        angle_rad_bottom = math.radians(skew_angle_bottom)

        # 计算新宽度
        total_offset_top = int(adjusted_top_half_height * math.tan(angle_rad_top))
        total_offset_bottom = int(bottom_half_height * math.tan(angle_rad_bottom))
        new_width = width + max(total_offset_top, total_offset_bottom)

        # 创建新的图像，初始化为透明
        new_img = Image.new('RGBA', (new_width, height))

        # 处理上半部分（平行四边形部分）
        for y in range(adjusted_top_half_height):
            # 计算当前行的偏移量，向右倾斜
            offset_top = int(y * math.tan(angle_rad_top))
            
            for x in range(width):
                # 计算新的x坐标，保持底边与下半部分重合
                new_x = x + total_offset_top - offset_top
                
                # 确保坐标在有效范围内
                if 0 <= new_x < new_width and 0 <= y < adjusted_top_half_height:
                    # 计算原图的y坐标，依据压缩或拉伸比例
                    original_y = int(y / top_half_height_ratio)
                    
                    # 确保原图的y坐标在有效范围内
                    if original_y < top_half_height:
                        new_img.putpixel((new_x, y), img.getpixel((x, original_y)))

        # 处理下半部分（保持上边不变，整体向左移动以与上半部分对齐）
        for y in range(bottom_half_height):
            # 计算当前行的偏移量，向左倾斜
            offset_bottom = int(y * math.tan(angle_rad_bottom))
            
            for x in range(width):
                # 计算新的x坐标，整体向左移动以与上半部分对齐
                new_x = x - offset_bottom
                
                # 确保坐标在有效范围内
                if 0 <= new_x < new_width and adjusted_top_half_height + y < height:
                    new_img.putpixel((new_x, adjusted_top_half_height + y), img.getpixel((x, top_half_height + y)))

        # 对处理后的图像进行平滑处理，减少锯齿效果
        new_img = new_img.filter(ImageFilter.SMOOTH)

        if isRGB == 1:
            return (pil2tensor(new_img.convert("RGB")),)
        
        return (pil2tensor(new_img),)

# 上半部分往右偏移下半部分也偏移节点
class SkewImageTopBottomRight(object):
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "skew_angle_top": ("INT", {
                    "default": 30,
                    "min": 0,
                    "max": 90,
                    "step": 1,
                    "display": 'number'
                }),
                "skew_angle_bottom": ("INT", {
                    "default": -15,
                    "min": -90,
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
                "isRGB": ("INT", {
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
    FUNCTION = "skew_image_top_bottom_right"
    
    # 上半部分往右偏移
    def skew_image_top_bottom_right(self, image, skew_angle_top=30, skew_angle_bottom=-15, bottom_half_height_ratio=0.5, isRGB=1):
        # 加载图像
        img = tensor2pil(image)[0].convert("RGBA")
        width, height = img.size

        # 计算下半部分的高度
        bottom_half_height = int(height * bottom_half_height_ratio)
        top_half_height = height - bottom_half_height

        # 将倾斜角度转换为弧度
        angle_rad_top = math.radians(skew_angle_top)
        angle_rad_bottom = math.radians(skew_angle_bottom)

        # 计算偏移量
        total_offset_top = int(top_half_height * math.tan(angle_rad_top))  # 上半部分向右的偏移
        total_offset_bottom = int(bottom_half_height * math.tan(angle_rad_bottom))  # 下半部分向左的偏移

        # 计算新宽度以适应下半部分的偏移
        new_width = width + total_offset_top + max(0, -total_offset_bottom)

        # 创建新的图像，初始化为透明
        new_img = Image.new('RGBA', (new_width, height))

        # 处理上半部分（平行四边形部分）
        for y in range(top_half_height):
            # 计算当前行的偏移量，向右倾斜
            offset_top = int(y * math.tan(angle_rad_top))

            for x in range(width):
                # 计算新的x坐标，向右偏移
                new_x = x + total_offset_top - offset_top
                
                # 确保坐标在有效范围内
                if 0 <= new_x < new_width and 0 <= y < top_half_height:
                    new_img.putpixel((new_x, y), img.getpixel((x, y)))

        # 处理下半部分（向左偏移）
        for y in range(bottom_half_height):
            # 计算当前行的偏移量，向左倾斜
            offset_bottom = int(y * math.tan(angle_rad_bottom))

            for x in range(width):
                # 计算新的x坐标，向左偏移
                new_x = x - offset_bottom
                
                # 确保坐标在有效范围内，动态调整图像宽度
                if 0 <= new_x < new_width and top_half_height + y < height:
                    new_img.putpixel((new_x, top_half_height + y), img.getpixel((x, top_half_height + y)))

        # 保存处理后的图像
        return (pil2tensor(new_img), )

# 旋转角度
class RotateImage(object):
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "angle": ("INT", {
                    "default": 30,
                    "min": -360,
                    "max": 360,
                    "step": 1,
                    "display": 'number'
                }),
                "isRGB": ("INT", {
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
    FUNCTION = "rotate_image"

    # 旋转角度
    def rotate_image(self, image, angle=30, isRGB=1):
        # 加载图像
        img = tensor2pil(image)[0].convert("RGBA")
        
        # 使用抗锯齿算法进行旋转，使用 Image.Resampling.LANCZOS 替代 Image.ANTIALIAS
        rotated_img = img.rotate(angle, expand=True, resample=Image.Resampling.BILINEAR)
        
        if isRGB == 1:
            return (pil2tensor(rotated_img.convert("RGB")), )
        
        # 保存处理后的图像
        return (pil2tensor(rotated_img), )

# resize图像的大小（图像会伸缩）
class ResizeImage(object):
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {
                    "default": 100,
                    "min": -360,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "height": ("INT", {
                    "default": 100,
                    "min": -360,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "isRGB": ("INT", {
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
    FUNCTION = "resize_image"
    
    # resize图像的大小（图像会伸缩）
    def resize_image(self, image, width=1000, height=1000, isRGB=1):
        # 加载图像
        img = tensor2pil(image)[0].convert("RGBA")
        resized_img = img.resize((width, height), 4)

        if isRGB == 1:
            return (pil2tensor(resized_img.convert("RGB")), )

        # 保存处理后的图像
        return (pil2tensor(resized_img), )

# resize图像的大小（图像保持纵横比）
class ResizeWithRatioImage(object):
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {
                    "default": 100,
                    "min": -360,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "height": ("INT", {
                    "default": 100,
                    "min": -360,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "isRGB": ("INT", {
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
    FUNCTION = "resize_with_ratio_image"
    
    # 上半部分往右偏移
    def resize_with_ratio_image(self, image, width=1000, height=1000, isRGB=1):
        # 加载图像
        img = tensor2pil(image)[0].convert("RGBA")
        img.thumbnail((width, height), 4)

        if isRGB == 1:
            return (pil2tensor(img.convert("RGB")), )

        # 保存处理后的图像
        return (pil2tensor(img), )

# 向四周扩图
class ExpandImage(object):
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT", {
                    "default": 100,
                    "min": -360,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "top": ("INT", {
                    "default": 100,
                    "min": -360,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "right": ("INT", {
                    "default": 100,
                    "min": -360,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "bottom": ("INT", {
                    "default": 100,
                    "min": -360,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "isRGB": ("INT", {
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
    FUNCTION = "expand_image"
    
    # 上半部分往右偏移
    def expand_image(self, image, left=1000, top=1000, right=1000, bottom=1000, isRGB=1):
        # 加载图像
        img = tensor2pil(image)[0].convert("RGBA")
        img = ImageOps.expand(img, border=(left, top, right, bottom), fill=(0, 0, 0, 0))

        if isRGB == 1:
            return (pil2tensor(img.convert("RGB")), )

        # 保存处理后的图像
        return (pil2tensor(img), )

# 清除图像四周的空白并调整图像大小
class ClearImageBorder(object):
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "tolerance": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "isRGB": ("INT", {
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
    FUNCTION = "clear_image_border"
    
    # 清除图像四周的空白并调整图像大小
    def clear_image_border(self, image, tolerance=0, isRGB=1):
        # 加载图像
        img = tensor2pil(image)[0].convert("RGBA")
        # 转换为灰度图像，获取 alpha 通道
        alpha = img.split()[3]
        
        # 计算裁剪的边界
        bbox = alpha.getbbox()  # 获取非透明区域的边界框
        
        if bbox is None:
            # 如果图像完全透明，返回空白图像
            result = Image.new("RGBA", img.size, (255, 255, 255, 0))
        else:
            # 考虑容差，扩大边界框
            left = max(0, bbox[0] - tolerance)
            upper = max(0, bbox[1] - tolerance)
            right = min(img.width, bbox[2] + tolerance)
            lower = min(img.height, bbox[3] + tolerance)

            # 使用计算后的边界裁剪图像
            result = img.crop((left, upper, right, lower))

        if isRGB == 1:
            return (pil2tensor(result.convert("RGB")), )

        # 保存处理后的图像
        return (pil2tensor(result), )

# 获取图像内空白区域尺寸
class GetImageBlankSize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tolerance": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT",)
    RETURN_NAMES = ("left", "top", "right", "bottom",)
    FUNCTION = "get_img_blank_size"

    CATEGORY = "img_process"

    def get_img_blank_size(self, image, tolerance=0):
        # 加载图像
        img = tensor2pil(image)[0].convert("RGBA")
        # 转换为灰度图像，获取 alpha 通道
        alpha = img.split()[3]

        # 计算裁剪的边界
        bbox = alpha.getbbox()  # 获取非透明区域的边界框

        if bbox is None:
            # 如果图像完全透明，返回空白图像
            return (img.width, img.height, img.width, img.height)

        # 考虑容差，扩大边界框
        left = max(0, bbox[0] - tolerance)
        upper = max(0, bbox[1] - tolerance)
        right = min(img.width, bbox[2] + tolerance)
        lower = min(img.height, bbox[3] + tolerance)

        # 返回四周空白像素数
        return (left, upper, img.width - right, img.height - lower)

# 计算非空白图像尺寸
class GetImageSizeWithoutBorder:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tolerance": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
            }
        }

    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("width", "height",)
    FUNCTION = "get_image_size_without_border"

    CATEGORY = "img_process"

    def get_image_size_without_border(self, image, tolerance=0):
        # 加载图像
        img = tensor2pil(image)[0].convert("RGBA")
        # 转换为灰度图像，获取 alpha 通道
        alpha = img.split()[3]

        # 计算裁剪的边界
        bbox = alpha.getbbox()  # 获取非透明区域的边界框

        if bbox is None:
            # 如果图像完全透明，返回空白图像
            return (0, 0)

        # 考虑容差，扩大边界框
        left = max(0, bbox[0] - tolerance)
        upper = max(0, bbox[1] - tolerance)
        right = min(img.width, bbox[2] + tolerance)
        lower = min(img.height, bbox[3] + tolerance)

        # 返回四周空白像素数
        return (right - left, lower - upper)

# 根据示例图像调整图像尺寸
class AdjustImageSize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "demo_image": ("IMAGE",),
                "tolerance": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "isRGB": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 1,
                    "step": 1,
                    "display": 'number'
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "adjust_image_size"

    CATEGORY = "img_process"

    def adjust_image_size(self, image, demo_image, tolerance=0, isRGB=1):
        # 加载图像
        img = tensor2pil(image)[0].convert("RGBA")
        demo_img = tensor2pil(demo_image)[0].convert("RGBA")

        # 转换为灰度图像，获取 alpha 通道
        demo_alpha = demo_img.split()[3]

        # 计算裁剪的边界
        demo_bbox = demo_alpha.getbbox()  # 获取非透明区域的边界框

        # 考虑容差，扩大边界框
        left = max(0, demo_bbox[0] - tolerance)
        upper = max(0, demo_bbox[1] - tolerance)
        right = min(demo_img.width, demo_bbox[2] + tolerance)
        lower = min(demo_img.height, demo_bbox[3] + tolerance)
        demo_inner_with, demo_inner_height = (right - left, lower - upper)

        print(right, left, lower, upper)
        # 计算非空白图像尺寸
        print(demo_inner_with, demo_inner_height)

        # 获取图像内空白区域尺寸
        demo_blank_left, demo_blank_top, demo_blank_right, demo_blank_bottom = GetImageBlankSize.get_img_blank_size(self, demo_image, tolerance)
        print(demo_blank_left, demo_blank_top, demo_blank_right, demo_blank_bottom)
        # 清除现有图案的空白区域
        # 转换为灰度图像，获取 alpha 通道
        alpha = img.split()[3]
        
        # 计算裁剪的边界
        bbox = alpha.getbbox()  # 获取非透明区域的边界框
        
        if bbox is None:
            # 如果图像完全透明，返回空白图像
            result = Image.new("RGBA", img.size, (255, 255, 255, 0))
        else:
            # 考虑容差，扩大边界框
            left = max(0, bbox[0] - tolerance)
            upper = max(0, bbox[1] - tolerance)
            right = min(img.width, bbox[2] + tolerance)
            lower = min(img.height, bbox[3] + tolerance)

            # 使用计算后的边界裁剪图像
            result = img.crop((left, upper, right, lower))
        # 调整现有图案和示例图案大小一样
        resized_img = result.resize((demo_inner_with, demo_inner_height), 4)
        # 调整现有图案的空白区域和示例图案一样
        resImg = ImageOps.expand(resized_img, border=(demo_blank_left, demo_blank_top, demo_blank_right, demo_blank_bottom), fill=(0, 0, 0, 0))

        if isRGB == 1:
            return (pil2tensor(resImg.convert("RGB")), )

        # 保存处理后的图像
        return (pil2tensor(resImg), )

# 图像分成两部分，上面一部分进行高度压缩，下面一部分不变
class SplitCompressTransform(object):
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "split_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "display": 'float'
                }),
                "compress_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "display": 'float'
                }),
                "isRGB": ("INT", {
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
    FUNCTION = "split_and_compress_image"

    # split_and_compress_image
    def split_and_compress_image(self, image, split_ratio=0.5, compress_ratio=0.5, isRGB=1):
        # 加载图像
        img = tensor2pil(image)[0].convert("RGBA")
        # 加载图像
        # 计算分割点
        split_point = int(img.height * split_ratio)

        # 分割图像
        top_half = img.crop((0, 0, img.width, split_point))
        bottom_half = img.crop((0, split_point, img.width, img.height))

        # 压缩上面一部分
        top_half = top_half.resize((top_half.width, int(top_half.height * compress_ratio)), 4)

        # 合并图像时图像大小也要改变
        new_height = top_half.height + bottom_half.height
        new_img = Image.new("RGBA", (img.width, new_height))

        new_img.paste(top_half, (0, 0))
        new_img.paste(bottom_half, (0, top_half.height))

        if isRGB == 1:
            return (pil2tensor(new_img.convert("RGB")), )

        # 保存处理后的图像
        return (pil2tensor(new_img), )

# 图像分成两部分，上下两部分都进行高度压缩
class SplitCompressTransformBoth(object):
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "split_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "display": 'float'
                }),
                "top_compress_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "display": 'float'
                }),
                "bottom_compress_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "display": 'float'
                }),
                "isRGB": ("INT", {
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
    FUNCTION = "split_and_compress_image"

    # split_and_compress_image
    def split_and_compress_image(self, image, split_ratio=0.5, top_compress_ratio=0.5, bottom_compress_ratio=0.5, isRGB=1):
        # 加载图像
        img = tensor2pil(image)[0].convert("RGBA")
        # 加载图像
        # 计算分割点
        split_point = int(img.height * split_ratio)

        # 分割图像
        top_half = img.crop((0, 0, img.width, split_point))
        bottom_half = img.crop((0, split_point, img.width, img.height))

        # 压缩上面一部分
        top_half = top_half.resize((top_half.width, int(top_half.height * top_compress_ratio)), 4)
        
        # 压缩下面一部分
        bottom_half = bottom_half.resize((bottom_half.width, int(bottom_half.height * bottom_compress_ratio)), 4)

        # 合并图像时图像大小也要改变
        new_height = top_half.height + bottom_half.height
        new_img = Image.new("RGBA", (img.width, new_height))

        new_img.paste(top_half, (0, 0))
        new_img.paste(bottom_half, (0, top_half.height))

        if isRGB == 1:
            return (pil2tensor(new_img.convert("RGB")), )

        # 保存处理后的图像
        return (pil2tensor(new_img), )

# 图像分成两部分，上面一部分进行高度压缩，下面一部分不变，去锯齿
class SplitCompressTransformNo(object):
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "split_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "display": 'float'
                }),
                "compress_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "display": 'float'
                }),
                "isRGB": ("INT", {
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
    FUNCTION = "split_and_compress_image_no"

    def split_and_compress_image_no(self, image, split_ratio=0.5, compress_ratio=0.5, isRGB=1):
        # 加载图像
        img = tensor2pil(image)[0].convert("RGBA")

        # 计算分割点
        split_point = int(img.height * split_ratio)

        # 分割图像
        top_half = img.crop((0, 0, img.width, split_point))
        bottom_half = img.crop((0, split_point, img.width, img.height))

        # 压缩上面一部分，使用高质量插值
        top_half = top_half.resize((top_half.width, int(top_half.height * compress_ratio)), Image.LANCZOS)

        # 合并图像时图像大小也要改变
        new_height = top_half.height + bottom_half.height
        new_img = Image.new("RGBA", (img.width, new_height))

        new_img.paste(top_half, (0, 0))
        new_img.paste(bottom_half, (0, top_half.height))

        return (pil2tensor(new_img), )

# 调整图像到指定尺寸（四周填充空白），默认图像在左上角，可设置offset
class ResizeImageOffset:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {
                    "default": 512,
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "offset_x": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "offset_y": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "resize_image_offset"

    CATEGORY = "img_process"

    def resize_image_offset(self, image, width, height, offset_x, offset_y):
        img = tensor2pil(image)[0].convert("RGBA")
        img_width, img_height = img.size

        new_width = width
        new_height = height

        # Check for out-of-bounds conditions
        if offset_x < 0 or offset_y < 0 or offset_x + img_width > new_width or offset_y + img_height > new_height:
            # Handle out-of-bounds: Crop the image to fit within the new dimensions.
            x1 = max(0, -offset_x)
            y1 = max(0, -offset_y)
            x2 = min(img_width, new_width - offset_x)
            y2 = min(img_height, new_height - offset_y)
            img = img.crop((x1, y1, x2, y2))
            # Recalculate dimensions after cropping
            img_width, img_height = img.size

        # Create new image and fill with background
        new_img = Image.new("RGBA", (new_width, new_height), (255, 255, 255, 0))

        # Paste the image, handling potential negative offsets (already handled by cropping).
        new_img.paste(img, (max(0, offset_x), max(0, offset_y)))  # Ensure positive offsets

        return (pil2tensor(new_img),)

# 调整通过点的位置调整图像
class ChangeRectangle:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {
                    "default": 512,
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "ori_x_1": ("INT", {
                    "default": 0,
                    "min": -100000,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "ori_y_1": ("INT", {
                    "default": 0,
                    "min": -100000,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "ori_x_2": ("INT", {
                    "default": 0,
                    "min": -100000,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "ori_y_2": ("INT", {
                    "default": 0,
                    "min": -100000,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "ori_x_3": ("INT", {
                    "default": 0,
                    "min": -100000,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "ori_y_3": ("INT", {
                    "default": 0,
                    "min": -100000,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "ori_x_4": ("INT", {
                    "default": 0,
                    "min": -100000,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "ori_y_4": ("INT", {
                    "default": 0,
                    "min": -100000,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "des_x_1": ("INT", {
                    "default": 0,
                    "min": -100000,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "des_y_1": ("INT", {
                    "default": 0,
                    "min": -100000,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "des_x_2": ("INT", {
                    "default": 0,
                    "min": -100000,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "des_y_2": ("INT", {
                    "default": 0,
                    "min": -100000,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "des_x_3": ("INT", {
                    "default": 0,
                    "min": -100000,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "des_y_3": ("INT", {
                    "default": 0,
                    "min": -100000,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "des_x_4": ("INT", {
                    "default": 0,
                    "min": -100000,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "des_y_4": ("INT", {
                    "default": 0,
                    "min": -100000,
                    "max": 100000,
                    "step": 1,
                    "display": 'number'
                }),
                "isRGB": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 1,
                    "step": 1,
                    "display": 'number'
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "change_rectangle"

    CATEGORY = "img_process"

    def change_rectangle(self, image, width, height, ori_x_1, ori_y_1, ori_x_2, ori_y_2, ori_x_3, ori_y_3, ori_x_4, ori_y_4,
    des_x_1, des_y_1, des_x_2, des_y_2, des_x_3, des_y_3, des_x_4, des_y_4, isRGB=1):
        # 将图像转换为 RGBA 格式
        pil_img = tensor2pil(image)[0].convert("RGBA")
        img = pil_to_cv2(pil_img)

        # 定义原始坐标和目标坐标
        original_points = np.float32([[ori_x_1, ori_y_1], [ori_x_2, ori_y_2], [ori_x_3, ori_y_3], [ori_x_4, ori_y_4]])  # 原始图像的四个角
        transformed_points = np.float32([[des_x_1, des_y_1], [des_x_2, des_y_2], [des_x_3, des_y_3], [des_x_4, des_y_4]])  # 变换后的四个角
        
        # 指定目标图像的大小
        output_size = (width, height)  # (宽, 高)

        # 计算透视变换矩阵
        matrix = cv2.getPerspectiveTransform(original_points, transformed_points)

        # 应用透视变换
        transformed_image = cv2.warpPerspective(img, matrix, output_size)

        res_pil_img = cv2_to_pil(transformed_image)

        if isRGB == 0:
            new_img = res_pil_img.convert("RGBA")

        if isRGB == 1:
            new_img = res_pil_img.convert("RGB")

        # Save the processed image
        return (pil2tensor(new_img), )

        # Save the processed image
        return (pil2tensor(cv2_to_pil(transformed_image)), )