from PIL import Image
import torch
from collections.abc import Callable
import numpy as np
import numpy.typing as npt

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
        img = tensor2pil(image)[0]
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