import numpy as np
import cv2
from tqdm import tqdm

def unsharp(input_name: str, output_name: str):
    image_name = input_name
    img = np.array(cv2.imread(image_name))
    image_h = len(img)
    image_w = len(img[0])
    mask_size = 3
    border = mask_size // 2
    new_img = np.zeros((image_w + border*2, image_h + border*2, 3))
    final_img = np.zeros((image_w, image_h, 3))

    new_img[border: -border, border: -border] = img
    mask = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])
    for i in tqdm(range(image_h)):
        for j in range(image_w):
            tmp = new_img[i:i+mask_size, j:j+mask_size]
            b = tmp[:, :, 0]
            g = tmp[:, :, 1]
            r = tmp[:, :, 2]
            b = b * mask
            g = g * mask
            r = r * mask
            final_img[i][j] = [int(b.sum()), int(g.sum()), int(r.sum())]
    cv2.imwrite(output_name, final_img)

def edge_detection(input_name: str, output_name: str):
    image_name = input_name
    img = np.array(cv2.imread(image_name, 0))
    image_h = len(img)
    image_w = len(img[0])
    mask_size = 3
    border = mask_size // 2
    new_img = np.zeros((image_w + border*2, image_h + border*2))
    horizontal_img = np.zeros((image_w, image_h))
    vertical_img = np.zeros((image_w, image_h))
    new_img[border: -border, border: -border] = img
    vertical_mask = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    horizontal_mask = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])
    for i in tqdm(range(image_h)):
        for j in range(image_w):
            tmp = new_img[i:i+mask_size, j:j+mask_size]
            after = tmp * vertical_mask
            vertical_img[i][j] = after.sum()
            after = tmp * horizontal_mask
            horizontal_img[i][j] = after.sum()
    final_img = np.sqrt(np.square(vertical_img) + np.square(horizontal_img))
    final_img = final_img * 255 // final_img.max()
    cv2.imwrite(output_name, final_img)

def main():
    image_name = 'Hw2image.jpg'
    output_name = 'Q4_unsharp.jpg'
    unsharp(image_name, output_name)
    image_name = 'Q4_unsharp.jpg'
    output_name = 'Q4_edge_detection.jpg'
    edge_detection(image_name, output_name)

if __name__ == '__main__':
    main()