import cv2
import os
import shutil
import numpy as np
from scipy import interpolate, signal
from matplotlib import pyplot as plt
import sys

# Global variables
cd = os.getcwd()  # get current directory
dir_path = rf"{cd}\Unmarked"
answer_path = rf"{cd}\Marked"


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# get_peak:                                                                   #
# Gets BW image and returns the highest black peak of intensity               #
#                                                                             #
# Returns:                                                                    #
#  * goodthd : highest black peak of intensity to be used by process_images() #
#  * params  : parameters to be used by plot_hist(savename, params)           #
#        - x_his: Array of x values of the image histogram                    #
#        - y_his: Array of y values of the image histogram                    #
#        - x_spl: Array of x values of the spline curve                       #
#        - y_spl: Array of y values of the spline curve                       #
#        - x_peaks: Array of x values of the found peaks                      #
#        - y_peaks: Array of y values of the found peaks                      #
#        - x_floors: Array of x values where threshold were set               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def get_peak(input_img):
    img_his = cv2.calcHist(input_img, [0], None, [256], [0, 256])

    y_his = img_his[:, 0]
    x_his = np.transpose(np.linspace(0, 255, 256))

    spl = interpolate.splrep(x_his, y_his, k=3, s=1000)

    n_samples = 256 * 20
    x_spl = np.linspace(0, 255, n_samples)
    y_spl = interpolate.splev(x_spl, spl).tolist()

    x_peaks = []
    x_floors = []

    y_peaks = []

    d = n_samples * 30 / 256  # Rule of 3: if n_samples == 256 then d_samples == 30
    ind_peaks = signal.find_peaks(y_spl, height=5., distance=d)[0]  # indexes where peaks can be found
    for i in range(len(ind_peaks)):
        ind = ind_peaks[i]  # index where i_th peak can be found
        y_peaks.append(y_spl[ind])  # Peaks on spline curve
        x_peaks.append(x_spl[ind])

        isfloor = ind

        while y_spl[isfloor] > 0:  # is floor (end of the mountain whose peak is y_spl[ind])
            isfloor += 1

        if isfloor - ind < (n_samples * 10 / 256):
            isfloor += (n_samples * 10 / 256)
            isfloor = int(isfloor)
        x_floors.append(x_spl[isfloor])

    goodthd = x_floors[y_peaks.index(
        max(y_peaks[:-1]))]  # Returns the x_floor corresponding to the highest peak (excluding white peak, i.e [:-1])
    return [goodthd, [x_his, y_his, x_spl, y_spl, x_peaks, y_peaks, x_floors]]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# plot_hist(savename, params):                                                #
# Plots histograms                                                            #
#                                                                             #
# Arguments:                                                                  #
#  * savename: base name of the source image (ex: 2Gy-28.jpg)                 #
#  * params  : parameters (returned by get_peak function)                     #
#        - x_his: Array of x values of the image histogram                    #
#        - y_his: Array of y values of the image histogram                    #
#        - x_spl: Array of x values of the spline curve                       #
#        - y_spl: Array of y values of the spline curve                       #
#        - x_peaks: Array of x values of the found peaks                      #
#        - y_peaks: Array of y values of the found peaks                      #
#        - x_floors: Array of x values where umbrals were set                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def plot_hist():
    unmarked_path = rf"{cd}\Unmarked"
    list_img = os.listdir(unmarked_path)

    for idx in range(len(list_img)):
        savename = list_img[idx]
        img_dir = dir_path + rf"\{savename}"

        # Image read
        img = cv2.imread(img_dir)

        # Image to BW
        img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, params = get_peak(img_bw)

        x_his = params[0]
        y_his = params[1]
        x_spl = params[2]
        y_spl = params[3]
        x_peaks = params[4]
        y_peaks = params[5]
        x_floors = params[6]

        plt.plot(x_his, y_his, color='gray', label='Histograma')
        # plt.scatter(x_peaks,y_his_r, color='black')                         # Para ver peaks sobre histograma
        if not 'wos' in sys.argv:
            plt.plot(x_spl, y_spl, color='blue', label='Curva spline')  # Add spline curve on plot
            plt.scatter(x_peaks, y_peaks, color='black')  # Add peaks on spline curve plot
            plt.vlines(x_floors, ymin=-10, ymax=500, linestyles='--', color='red',
                       label='Umbrales de Intensidad')  # Add thresholds on plot

        plt.xlabel('Intensidad de iluminacion')
        plt.ylabel('Cantidad de píxeles')

        plotname = savename.split('.')[0]
        plt.title(f'{plotname}')
        plt.legend()
        plt.savefig(plotname + "_hist.jpg")
        plt.close()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# process_images:                                                             #
# Gets raw images from /Unmarked folder and processes them into BW, then      #
# binary, then morphological transform and saves the results                  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def process_images():
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
        thd = get_peak(img_bw)[0]  # Gets highest black peak

        for ad in range(len(additions)):

            _, img_bin = cv2.threshold(img_bw, thd + additions[f'A{ad}'], 255, cv2.THRESH_BINARY)
            for m in range(len(morphs)):
                for k in range(len(kernels)):

                    img_p = cv2.morphologyEx(img_bin, morphs[f'M{m}'], kernels[f'K{k + 2}'])

                    if m == 0:
                        op = 'C'
                    else:
                        op = 'O'

                    if 's' in sys.argv:
                        ad_txt = additions[f'A{ad}']
                        if ad_txt == 0:
                            ad_txt = '00'
                        cv2.imwrite(save_name + f"_P({op}{k + 2}a{ad_txt}).jpg", img_p)
                    else:
                        cv2.imwrite(save_name + f"_P({op}{k + 2}).jpg", img_p)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# images_to_processed:                                                        #
# Moves images from global folder into Processed folder. It also classifies   #
# them into different categories according to previous transformations        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def images_to_processed():
    dst_path = rf"{cd}\Processed_spline"
    additions = True
    file_list = os.listdir(cd)
    if 'n' in sys.argv:
        dst_path = rf"{cd}\Processed_otsu"
    img_list = []

    # Remove non-images files from file_list
    for idx in range(len(file_list)):
        if '.jpg' in file_list[idx]:
            img_list.append(file_list[idx])

    # Locate every image
    if 's' in sys.argv:
        ind1 = -10  # Refers to morphological filter (O/C)
        ind2 = -9   # Refers to kernel size (2/3/4/5)
        ind3 = -7   # Refers to image addition (00,10,20,30)
    else:
        ind1 = -7   # Refers to morphological filter (O/C)
        ind2 = -6   # Refers to kernel size (2/3/4/5)


    for idx in range(len(img_list)):
        img = cd + rf"\{img_list[idx]}"

        # Opened images transfer according to img category
        if img[ind1] == 'O':
            for i in range(2, 6):
                if img[ind2] == str(i):
                    if 's' in sys.argv:
                        for j in range(4):
                            if img[ind3] == str(j):
                                new_path = dst_path + rf"\Open {i}\Addition {j}0"
                                if not os.path.isdir(new_path):
                                    os.makedirs(new_path)
                                if not os.path.isfile(new_path + rf"\{img_list[idx]}"):
                                    shutil.move(img, new_path)
                    else:
                        new_path = dst_path + rf"\Open {i}"
                        if not os.path.isdir(new_path):
                            os.makedirs(new_path)
                        if not os.path.isfile(new_path + rf"\{img_list[idx]}"):
                            shutil.move(img, new_path)







        # Closed images transfer according to img category
        elif img[ind1] == 'C':
            for i in range(2, 6):
                if img[ind2] == str(i):
                    if 's' in sys.argv:
                        for j in range(4):
                            if img[ind3] == str(j):
                                new_path = dst_path + rf"\Close {i}\Addition {j}0"
                                if not os.path.isdir(new_path):
                                    os.makedirs(new_path)
                                if not os.path.isfile(new_path + rf"\{img_list[idx]}"):
                                    shutil.move(img, new_path)
                    else:
                        new_path = dst_path + rf"\Close {i}"
                        if not os.path.isdir(new_path):
                            os.makedirs(new_path)
                        if not os.path.isfile(new_path + rf"\{img_list[idx]}"):
                            shutil.move(img, new_path)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# images_to_hist:                                                             #
# Moves images from global folder into Histogram folder. It also classifies   #
# them into different categories according to previous transformations        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def images_to_hist():
    dst_path = rf"{cd}\Histograms"
    if 'wos' in sys.argv:
        dst_path += '\wo spline'
    else:
        dst_path += '\w spline'
    if not os.path.isdir(dst_path):
        os.makedirs(dst_path)
    file_list = os.listdir(cd)
    img_list = []

    # Saves in img_list all files from file_list which are images
    for idx in range(len(file_list)):
        if '.jpg' in file_list[idx]:
            img_list.append(file_list[idx])

    for idx in range(len(img_list)):
        img = cd + rf"\{img_list[idx]}"
        shutil.move(img, dst_path)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  Execution                                                                        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  Default Configuration                                                            #
# > process.py    : Processes images and stores them in Processed (method: spline)  #
# > process.py n   : Processes images and stores them in Processed (method: otsu)   #
# > process.py p  : Forces execution of default configuration                       #
#                                                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                                   #
# > process.py c        : Cuts chromosomes, stores them in Chromosomes/images and   #
#                         creates labels.txt                                        #
# > process.py h        : Gets histograms plots (with spline curve, by default)     #
# > process.py h wos    : Gets histograms plots without spline curve                #
#                                                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# > main.py
if len(sys.argv) == 1 or 'p' in sys.argv or (len(sys.argv) == 2 and 'n' in sys.argv):
    process_images()
    images_to_processed()

# > main.py h
if 'h' in sys.argv and 'd' not in sys.argv:
    plot_hist()
    images_to_hist()


