import numpy as np
import cv2
from tqdm import tqdm

def get_mask(size: int, sigma: float):
    minX = minY = -(size // 2)
    maxX = maxY = (size // 2)
    x, y = np.mgrid[minX : maxX + 1, minY : maxY + 1]
    kernel = np.exp((-(x**2 + y**2))/(2 * (sigma ** 2)))
    kernel = kernel / kernel.sum()
    return kernel

def gaussian_blur(input_name: str, output_name: str, mask_size: int, sigma: float):
    image_name = input_name
    img = np.array(cv2.imread(image_name))
    image_h = len(img)
    image_w = len(img[0])
    border = mask_size // 2
    new_img = np.zeros((image_w + border*2, image_h + border*2, 3))
    final_img = np.zeros((image_w, image_h, 3))
    new_img[border: -border, border: -border] = img
    blur_mask = get_mask(mask_size, sigma)
    for i in tqdm(range(image_h)):
        for j in range(image_w):
            tmp = new_img[i:i+mask_size, j:j+mask_size]
            b = tmp[:,:,0]
            g = tmp[:,:,1]
            r = tmp[:,:,2]
            b = b * blur_mask
            g = g * blur_mask
            r = r * blur_mask
            final_img[i][j] = [int(b.sum()), int(g.sum()), int(r.sum())]
    cv2.imwrite(output_name, final_img)

def get_PSNR(input_name: str, output_name: str) -> float:
    return cv2.PSNR(cv2.imread(input_name), cv2.imread(output_name))

def main():
    image_name = 'Hw2image.jpg'
    # Hw2 Q1, all sigma is 0.5
    size_list = [3, 7, 11]
    for i in size_list:
        gaussian_blur(image_name, f'Q1_{i}x{i}.jpg', i, 0.5)
        print(f'Q1_{i}x{i}.jpg PSNR is', get_PSNR(image_name, f'Q1_{i}x{i}.jpg'))

    # Hw2 Q2, all size = 3x3
    sigma_list = [1, 10, 30]
    for i in sigma_list:
        gaussian_blur(image_name, f'Q2_sigma_{i}.jpg', 3, i)
        print(f'Q2_sigma_{i}.jpg PSNR is', get_PSNR(image_name, f'Q2_sigma_{i}.jpg'))
if __name__ == '__main__':
    main()