import numpy as np 
import cv2
from scipy.ndimage.interpolation import rotate

def flip(image, axis):
    if axis == 0:
        return image[::-1, :, :]
    elif axis == 1:
        return image[:, ::-1, :]
    
    else:
        raise Exception('error axis')


def random_rotation(image, roate_angle=(0, 180)):
    a = np.random.randint(*roate_angle)
    return rotate(image, a)



def gaussian_blur(image, kernel_size=None, sigma=None):
    if not kernel_size:
        kernel_size = (image.shape[0] // 10) // 2 * 2 + 1

    if not sigma:
        sigma = cv2.BORDER_DEFAULT

    return cv2.GaussianBlur(
        image, 
        (kernel_size, kernel_size),
        sigma
    )
    

def random_blur(image, p=1.0):
    kernel_size = (image.shape[0] // 10) // 2 * 2 + 1
    sigma = np.random.rand() * 2
    
    if np.random.rand() < p:
        return gaussian_blur(image, kernel_size, sigma=sigma)
        
    return image

