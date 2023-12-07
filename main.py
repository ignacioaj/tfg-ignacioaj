import cv2
import os
import numpy as np
from functions import getPeak

kernels = {
    "K2" : np.ones((2,2), np.uint8),
    "K3" : np.ones((3,3), np.uint8),
    "K4" : np.ones((4,4), np.uint8),
    "K5" : np.ones((5,5), np.uint8)
}

k10 = np.ones((5,5), np.uint8)

morphs = {
    "M0" : cv2.MORPH_CLOSE,
    "M1" : cv2.MORPH_OPEN
}

additions = {
    "A0" : 00,
    "A1" : 10,
    "A2" : 20,
    "A3" : 30
}

#focus_kernel = np.matrix([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

dir_path = rf"\Unmarked"
answer_path = rf"\Marked"
list_img = os.listdir(dir_path)
list_ans = os.listdir(answer_path)

for idx in range(len(list_img)):
    img_dir = dir_path + rf"\{list_img[idx]}"
    img_ans_dir = answer_path + rf"\{list_ans[idx]}"
    save_name = list_img[idx].split('.')[0]

    #Image read
    img = cv2.imread(img_dir)
    img_ans = cv2.imread(img_ans_dir)

    #Image to BW
    img_bw    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_bwans = cv2.cvtColor(img_ans, cv2.COLOR_BGR2GRAY)

    # Image to Binary

    thd = getPeak(img_bw)  # Last peak is neglected bcs it corresponds to white pixels
    _, black = cv2.threshold(img_bwans, 255, 255, cv2.THRESH_BINARY)

    for ad in range(len(additions)):

        _, img_bin = cv2.threshold(img_bw, thd+additions[f'A{ad}'], 255, cv2.THRESH_BINARY)
        for m in range(len(morphs)):
            for k in range(len(kernels)):

                img_p = cv2.morphologyEx(img_bin, morphs[f'M{m}'], kernels[f'K{k+2}'])

                # mosaic = np.vstack([mosaic1, mosaic2])
                if m == 0 :
                    op = 'C'
                else:
                    op = 'O'

                ad_txt = additions[f'A{ad}']
                if ad_txt == 0:
                    ad_txt = '00'
                cv2.imwrite(save_name + f"_P(a{ad_txt}{op}{k+2}).jpg", img_p)