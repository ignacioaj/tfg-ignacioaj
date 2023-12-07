import cv2
import os
import numpy as np
from scipy import interpolate, signal

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# getPeaks:                                                                   #
# Gets BW image and returns the highest black peak of intensity               #
# DEPRICATED: Gets BW image and returns global maximums (peaks) of intensity  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def getPeak(input_img):

    img_his = cv2.calcHist(input_img,[0],None, [256],[0,256])

    y_his = img_his[:,0]
    x_his = np.transpose(np.linspace(0,255,256))

    spl = interpolate.splrep(x_his, y_his,k=3,s=1000)

    n_samples = 256*20
    x2 = np.linspace(0, 255, n_samples)
    y2 = interpolate.splev(x2, spl).tolist()

    x_peaks = []
    x_floors = []

    y_peaks = []

    d = n_samples * 30 / 256
    ind_peaks = signal.find_peaks(y2, height=5, distance=d)[0]  # indexes where peaks can be found
    for i in range(len(ind_peaks)):
        ind = ind_peaks[i]  # index where i_th peak can be found
        y_peaks.append(y2[ind])  # Peaks on spline curve
        x_peaks.append(x2[ind])

        isfloor = ind

        while y2[isfloor] > 0:   #is floor (end of the mountain whose peak is y2[ind])
            isfloor += 1

        if isfloor - ind < (n_samples * 10 / 256):
            isfloor += (n_samples * 10 / 256)
            isfloor = int(isfloor)
        x_floors.append(x2[isfloor])

    return x_floors[y_peaks.index(max(y_peaks[:-1]))] #Returns the x_floor corresponding to the highest peak (excluding white peak, i.e [:-1])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# processImages:                                                              #
# Gets raw images from /Unmarked folder and processes them into BW, then      #
# binary, then morphological transform and saves the results                  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def processImages():
    kernels = {
        "K2": np.ones((2, 2), np.uint8),
        "K3": np.ones((3, 3), np.uint8),
        "K4": np.ones((4, 4), np.uint8),
        "K5": np.ones((5, 5), np.uint8)
    }

    k10 = np.ones((5, 5), np.uint8)

    morphs = {
        "M0": cv2.MORPH_CLOSE,
        "M1": cv2.MORPH_OPEN
    }

    additions = {
        "A0": 00,
        "A1": 10,
        "A2": 20,
        "A3": 30
    }

    # focus_kernel = np.matrix([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    cd = os.getcwd()  # get current directory
    dir_path = rf"{cd}\Unmarked"
    answer_path = rf"{cd}\Marked"
    list_img = os.listdir(dir_path)
    list_ans = os.listdir(answer_path)

    for idx in range(len(list_img)):
        img_dir = dir_path + rf"\{list_img[idx]}"
        img_ans_dir = answer_path + rf"\{list_ans[idx]}"
        save_name = list_img[idx].split('.')[0]

        # Image read
        img = cv2.imread(img_dir)
        img_ans = cv2.imread(img_ans_dir)

        # Image to BW
        img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_bwans = cv2.cvtColor(img_ans, cv2.COLOR_BGR2GRAY)

        # Image to Binary

        thd = getPeak(img_bw)  # Last peak is neglected bcs it corresponds to white pixels
        _, black = cv2.threshold(img_bwans, 255, 255, cv2.THRESH_BINARY)

        for ad in range(len(additions)):

            _, img_bin = cv2.threshold(img_bw, thd + additions[f'A{ad}'], 255, cv2.THRESH_BINARY)
            for m in range(len(morphs)):
                for k in range(len(kernels)):

                    img_p = cv2.morphologyEx(img_bin, morphs[f'M{m}'], kernels[f'K{k + 2}'])

                    # mosaic = np.vstack([mosaic1, mosaic2])
                    if m == 0:
                        op = 'C'
                    else:
                        op = 'O'

                    ad_txt = additions[f'A{ad}']
                    if ad_txt == 0:
                        ad_txt = '00'
                    cv2.imwrite(save_name + f"_P(a{ad_txt}{op}{k + 2}).jpg", img_p)

#Execution--
processImages()