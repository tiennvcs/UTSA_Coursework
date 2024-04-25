from PIL import Image, ImageFilter
import numpy as np


def rescale(input_img, desired_size: np.ndarray) -> np.ndarray:
    return input_img.resize(desired_size)
     
def blur_edge(input_img, desired_size: np.ndarray):
    # blur radius and diameter
    radius, diameter = 10, 20
    # Paste image on white background
    background_size = (input_img.size[0] + diameter, input_img.size[1] + diameter)
    background = Image.new('RGB', background_size, (255, 255, 255))
    background.paste(input_img, (radius, radius))
    # create new images with white and black
    mask_size = (input_img.size[0] + diameter, input_img.size[1] + diameter)
    mask = Image.new('L', mask_size, 255)
    black_size = (input_img.size[0] - diameter, input_img.size[1] - diameter)
    black = Image.new('L', black_size, 0)
    # create blur mask
    mask.paste(black, (diameter, diameter))
    # Blur image and paste blurred edge according to mask
    blur = background.filter(ImageFilter.GaussianBlur(radius / 2))
    background.paste(blur, mask=mask)    
    return background.crop(((78-64)//2, (78-64)//2, (78+64)//2, (78+64)//2))

# def blur_edge2(src, dst):
#     src = np.asarray(src)
#     dst = np.asarray(dst)
#     out = np.empty(src.shape, dtype = 'float')
#     alpha = np.index_exp[:, :, 3:]
#     rgb = np.index_exp[:, :, :3]
#     src_a = src[alpha]/255.0
#     dst_a = dst[alpha]/255.0
#     out[alpha] = src_a+dst_a*(1-src_a)
#     old_setting = np.seterr(invalid = 'ignore')
#     out[rgb] = (src[rgb]*src_a + dst[rgb]*dst_a*(1-src_a))/out[alpha]
#     np.seterr(**old_setting)    
#     out[alpha] *= 255
#     np.clip(out,0,255)
#     # astype('uint8') maps np.nan (and np.inf) to 0
#     out = out.astype('uint8')
#     out = Image.fromarray(out, 'RGBA')
#     return out


def paste(source_img, patch_img):
    back_im = source_img.copy()
    random_loc = tuple(np.random.randint(224-64, size=2))
    back_im.paste(patch_img, random_loc)
    return back_im