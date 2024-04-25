import os
import numpy as np
from read_write import load_image, save_img
from img_manipulation import rescale, blur_edge, paste
from torchvision.transforms import v2


class PatchTrojans:

    def __init__(self, trojan_file: str):
        self._source_img_size = (224, 224)
        self._trojan_img_size = (64, 64)
        self._output_dir = "./output/test/patch_trojan"
        
        self._trojan_img = load_image(trojan_file)
        self._rescaled_trojan_img = rescale(self._trojan_img, self._trojan_img_size)
        
        self._jitter = v2.ColorJitter(brightness=.5, hue=.3)
        self._blurrer = v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))
    
    def _color_jitter(self, input_img):
        return self._jitter(input_img)
    
    def _gaussian_noise(self, input_img):
        return self._blurrer(input_img)
        
    def _blur_edge(self, input_img):
        return blur_edge(input_img, self._trojan_img_size)
    
    def implant(self, source_img, debug=False):
        resized_src_img = rescale(source_img, self._source_img_size)
        jittered_img = self._color_jitter(self._rescaled_trojan_img)
        blurred_img = self._gaussian_noise(jittered_img)        
        blur_edge_img = self._blur_edge(blurred_img)
        implanted_img = paste(resized_src_img, blur_edge_img)

        if debug:
            save_img(os.path.join(self._output_dir, 'original_source_img.png'), source_img)
            save_img(os.path.join(self._output_dir, 'original_trojan_img.png'), self._trojan_img)
            save_img(os.path.join(self._output_dir, 'rescaled_trojan_img.png'), self._rescaled_trojan_img)
            save_img(os.path.join(self._output_dir, 'rescaled_source_img.png'), resized_src_img)
            save_img(os.path.join(self._output_dir, 'jittered_trojan_img.png'), jittered_img)
            save_img(os.path.join(self._output_dir, 'blurred_trojan_img.png'), blurred_img)
            save_img(os.path.join(self._output_dir, 'blur_edge_trojan_img.png'), blur_edge_img)
            save_img(os.path.join(self._output_dir, 'implanted_img.png'), implanted_img)


if __name__ == '__main__':
    trojan_file = "/home/tiennv/Github/UTSA_Coursework/Spring2024/CS6283_DL/Project/trojan_implantation/data/images/patch_trojan/smile.png"
    pt = PatchTrojans(trojan_file=trojan_file)

    source_file = "/home/tiennv/Github/UTSA_Coursework/Spring2024/CS6283_DL/Project/trojan_implantation/data/images/patch_trojan/source.jpg"

    source_img = load_image(source_file)
    

    pt.implant(source_img=source_img, debug=True)