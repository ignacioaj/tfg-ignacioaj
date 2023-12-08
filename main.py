import cv2
import os
import shutil
import numpy as np
from scipy import interpolate, signal
from PIL import Image

# Global variables
cd = os.getcwd()  # get current directory
dir_path = rf"{cd}\Unmarked"
answer_path = rf"{cd}\Marked"


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# get_peak:                                                                   #
# Gets BW image and returns the highest black peak of intensity               #
# DEPRICATED: Gets BW image and returns global maximums (peaks) of intensity  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def get_peak(input_img):
    img_his = cv2.calcHist(input_img, [0], None, [256], [0, 256])

    y_his = img_his[:, 0]
    x_his = np.transpose(np.linspace(0, 255, 256))

    spl = interpolate.splrep(x_his, y_his, k=3, s=1000)

    n_samples = 256 * 20
    x2 = np.linspace(0, 255, n_samples)
    y2 = interpolate.splev(x2, spl).tolist()

    x_peaks = []
    x_floors = []

    y_peaks = []

    d = n_samples * 30 / 256
    ind_peaks = signal.find_peaks(y2, height=5, distance=d)[0]  # indexes where peaks can be found
    for i in range(len(ind_peaks)):
        ind = ind_peaks[i]       # index where i_th peak can be found
        y_peaks.append(y2[ind])  # Peaks on spline curve
        x_peaks.append(x2[ind])

        isfloor = ind

        while y2[isfloor] > 0:   # is floor (end of the mountain whose peak is y2[ind])
            isfloor += 1

        if isfloor - ind < (n_samples * 10 / 256):
            isfloor += (n_samples * 10 / 256)
            isfloor = int(isfloor)
        x_floors.append(x2[isfloor])

    return x_floors[y_peaks.index(
        max(y_peaks[:-1]))]  # Returns the x_floor corresponding to the highest peak (excluding white peak, i.e [:-1])


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
        img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_bwans = cv2.cvtColor(img_ans, cv2.COLOR_BGR2GRAY)

        # Image to Binary

        thd = get_peak(img_bw)  # Last peak is neglected bcs it corresponds to white pixels
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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# images_to_processed:                                                        #
# Moves images from global folder into Processed folder. It also classifies   #
# them into different categories according to previous transformations        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def images_to_processed():
    dst_path = rf"{cd}\Processed"
    file_list = os.listdir(cd)
    img_list = []

    # Remove non-images files from file_list
    for idx in range(len(file_list)):
        if '.jpg' in file_list[idx]:
            img_list.append(file_list[idx])

    # Locate every image
    for idx in range(len(img_list)):

        img = cd + rf"\{img_list[idx]}"

        # Opened images transfer according to img category
        if img[-7] == 'O':
            for i in range(2, 6):
                if img[-6] == str(i):
                    for j in range(4):
                        if img[-9] == str(j):
                            new_path = dst_path + rf"\Open {i}\Addition {j}0"
                            if not os.path.isdir(new_path):
                                os.makedirs(new_path)
                            if not os.path.isfile(new_path + rf"\{img_list[idx]}"):
                                shutil.move(img, new_path)

        # Closed images transfer according to img category
        elif img[-7] == 'C':
            for i in range(2, 6):
                if img[-6] == str(i):
                    for j in range(4):
                        if img[-9] == str(j):
                            new_path = dst_path + rf"\Close {i}\Addition {j}0"
                            if not os.path.isdir(new_path):
                                os.makedirs(new_path)
                            if not os.path.isfile(new_path + rf"\{img_list[idx]}"):
                                shutil.move(img, new_path)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# chromosome_cut:                                                             #
# Gets every untransformed unmarked image and extract every single chromosome #
# Also, overwrites file label.txt with the chr name and category (0nor-1dic)  #
# Also, prints: number of nor, dic and nor+dic, as well as respective ratios  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def chromosome_cut():
    tag_path = rf"{cd}\Labels"
    save_labeldir = rf"{cd}\Chromosomes\labels.txt"
    img_list = os.listdir(dir_path)
    name_list = []
    for idx in range(len(img_list)):
        name_list.append(img_list[idx].split('.')[0])

    dic = 0
    nor = 0
    w_list = []
    wmax = 0
    h_list = []
    hmax = 0
    new_txt = ''

    for idx in range(len(name_list)):
        img_dir = rf"{dir_path}\{img_list[idx]}"
        img = Image.open(img_dir)
        imgsh = cv2.imread(img_dir)
        height = imgsh.shape[0]
        width = imgsh.shape[1]

        with open(rf"{tag_path}\{name_list[idx]}.txt", 'r') as txt:
            lines = txt.readlines()
            cnt = 0
            for l in lines:
                args = l.split(' ')
                args[-1] = args[-1][:-2]
                xrc = float(args[1]);
                yrc = float(args[2]);
                rw = float(args[3]);
                rh = float(args[4])
                xc = width * xrc;
                yc = height * yrc  # x,y of the center of the bbox
                wb = rw * width;
                hb = rh * height  # height,width of the bbox
                x1 = round(xc - wb / 2);
                x2 = round(xc + wb / 2)
                y1 = round(yc - hb / 2);
                y2 = round(yc + hb / 2)

                cnt += 1
                w_list.append(wb)
                if wb == max(w_list): wmax = wb
                h_list.append(hb)
                if wb == max(w_list): hmax = hb

                if args[0] == '0':
                    nor += 1
                else:
                    dic += 1

                new_txt += f'{name_list[idx]}_c{cnt} {args[0]}\n'

                chromosome = img.crop((x1, y1, x2, y2))
                chromosome.save(f"{name_list[idx]}_c{cnt}.jpg")

    if not os.path.isfile(save_labeldir):
        with open(save_labeldir, "a") as label:
            label.write(new_txt)

    print(f"Cantidad total de cromosomas       : {dic + nor}")
    print(f"Cantidad de cromosomas dicentricos : {dic}")
    print(f"Cantidad de cromosomas normales    : {nor}")
    print(f"Ratio de cromosomas dicentricos    : {dic / (dic + nor) * 100} %")
    print(f"Ratio de cromosomas normales       : {nor / (dic + nor) * 100} %")
    print(f"Máxima altura de bbox              : {round(hmax)} px")
    print(f"Máxima anchura de bbox             : {round(wmax)} px")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# images_to_chro:                                                             #
# Moves images from global folder into Processed folder. It also classifies   #
# them into different categories according to previous transformations        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def images_to_chro():
    folder = rf"\Chromosomes\images"
    dst_path = cd + folder
    file_list = os.listdir(cd)
    img_list = []

    # Remove non-images files from file_list
    for idx in range(len(file_list)):
        if '.jpg' in file_list[idx]:
            img_list.append(file_list[idx])

    for idx in range(len(img_list)):
        img = cd + rf"\{img_list[idx]}"
        if not os.path.isdir(dst_path):
            os.makedirs(dst_path)
        if not os.path.isfile(dst_path+rf"/{img_list[idx]}"):
            shutil.move(img, dst_path)


# Execution--
process_images()
images_to_processed()
chromosome_cut()
images_to_chro()