from PIL import Image
import numpy as np

def keep_asR_resize(image):
    
    W, H = image.size
    aspect_list = np.array([12/3, 9/4, 6/6])
    size_list = np.array([[12, 3], [9, 4], [6, 6]]) * 32
    #Using below ones might be better.
    #aspect_list = np.array([36/1, 18/2, 12/3, 9/4, 6/6])
    #size_list = np.array([[36, 1], [18, 2], [12, 3], [9, 4], [6, 6]]) * 32
    if W > H:
        region_aspect = W/(H+1e-15)
        new_w, new_h = size_list[np.argmin(np.absolute(aspect_list - region_aspect))]
    else:
        region_aspect = H/(W+1e-15)
        new_h, new_w = size_list[np.argmin(np.absolute(aspect_list - region_aspect))]
        
    resize_shape = (new_w, new_h)
    image = image.resize(resize_shape, Image.ANTIALIAS)
    
    return image, resize_shape
