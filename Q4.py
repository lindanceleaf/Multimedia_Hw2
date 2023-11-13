import numpy as np
import cv2
from tqdm import tqdm

def convolution(input_name: str, output_name: str):
    image_name = input_name
    img = np.array(cv2.imread(image_name, 0))
    image_h = len(img)
    image_w = len(img[0])
    mask_size = 3
    border = mask_size // 2
    new_img = np.zeros((image_w + border*2, image_h + border*2))
    final_img = np.zeros((image_w, image_h))

    new_img[border: -border, border: -border] = img
    mask = [
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ]
    for i in tqdm(range(image_h)):
        for j in range(image_w):
            tmp1 = new_img[i:i+mask_size, j:j+mask_size]
            after = tmp1 * mask
            final_img[i][j] = after.sum()
    cv2.imwrite(output_name, final_img)

def main():
    image_name = 'Hw2image.jpg'
    output_name = 'Q4.jpg'
    convolution(image_name, output_name)

if __name__ == '__main__':
    main()