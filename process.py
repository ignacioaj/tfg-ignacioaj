import cv2
import os
import shutil
import numpy as np
from scipy import interpolate, signal
from matplotlib import pyplot as plt
from PIL import Image
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
# plot_hist(savename, params):                                                 #
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

                    # mosaic = np.vstack([mosaic1, mosaic2])
                    if m == 0:
                        op = 'C'
                    else:
                        op = 'O'

                    ad_txt = additions[f'A{ad}']
                    if ad_txt == 0:
                        ad_txt = '00'
                    cv2.imwrite(save_name + f"_P({op}{k + 2}a{ad_txt}).jpg", img_p)


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
    ind1 = -10  # Refers to morphological filter (O/C)
    ind2 = -9   # Refers to kernel size (2/3/4/5)
    ind3 = -7   # Refers to image addition (00,10,20,30)
    for idx in range(len(img_list)):
        img = cd + rf"\{img_list[idx]}"

        # Opened images transfer according to img category
        if img[ind1] == 'O':
            for i in range(2, 6):
                if img[ind2] == str(i):
                    for j in range(4):
                        if img[ind3] == str(j):
                            new_path = dst_path + rf"\Open {i}\Addition {j}0"
                            if not os.path.isdir(new_path):
                                os.makedirs(new_path)
                            if not os.path.isfile(new_path + rf"\{img_list[idx]}"):
                                shutil.move(img, new_path)

        # Closed images transfer according to img category
        elif img[ind1] == 'C':
            for i in range(2, 6):
                if img[ind2] == str(i):
                    for j in range(4):
                        if img[ind3] == str(j):
                            new_path = dst_path + rf"\Close {i}\Addition {j}0"
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
                pars = l.split(' ')
                pars[-1] = pars[-1][:-2]
                xrc = float(pars[1]);
                yrc = float(pars[2]);
                rw = float(pars[3]);
                rh = float(pars[4])
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

                if pars[0] == '0':
                    nor += 1
                else:
                    dic += 1

                new_txt += f'{name_list[idx]}_c{cnt} {pars[0]}\n'

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
        if not os.path.isfile(dst_path + rf"/{img_list[idx]}"):
            shutil.move(img, dst_path)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# delete_images()                                                             #
# Deletes images from main directory (but not from its folders)               #
# To delete images from Histograms you should use function delete_hist()      #
# To delete images from Chromosomes (also txt) you should use delete_chro()   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def delete_images():
    list_file = os.listdir(cd)

    for idx in range(len(list_file)):
        file = cd + rf"\{list_file[idx]}"
        if 'jpg' in file:
            os.remove(file)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# delete_proc()                                                               #
# Deletes images from folder 'Processed'                                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def delete_proc():
    folder_list = []
    dir_path = ""
    additions = True
    if 's' in sys.argv and not 'n' in sys.argv:
        dir_path = rf"{cd}\Processed_spline"
        folder_list  = os.listdir(dir_path)
    elif 'n' in sys.argv and not 's' in sys.argv:
        dir_path(rf"{cd}\Processed_otsu")
        folder_list  = os.listdir(dir_path)
        additions = False

    for folder in range(len(folder_list)):


        if additions:
            folder2_list = os.listdir(rf"{dir_path}\{folder_list[folder]}")
            for folder2 in range(len(folder2_list)):
                file_list = os.listdir(rf"{dir_path}\{folder_list[folder]}\{folder2_list[folder2]}")
                for idx in range(len(file_list)):
                    file = rf"{dir_path}\{folder_list[folder]}\{folder2_list[folder2]}\{file_list[idx]}"
                    if 'jpg' in file:

                        os.remove(file)
        else:
            file_list = os.listdir(dir_path + rf"\{folder_list[folder]}")
            for idx in range(len(file_list)):
                file = dir_path + rf"{folder_list[folder]}\{file_list[idx]}"
                if 'jpg' in file:
                    os.remove(file)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# delete_hist()                                                               #
# Deletes images from folder 'Histograms'                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def delete_hist():
    his_path = rf"{cd}\Histograms"
    list_folders = os.listdir(his_path)

    if 'wos' not in sys.argv and not 'w' not in sys.argv:
        for folder in range(len(list_folders)):
            folder_dir = his_path + rf"\{list_folders[folder]}"
            list_hist = os.listdir(folder_dir)
            for img in range(len(list_hist)):
                if 'jpg' in list_hist[img]:
                    hist = folder_dir + rf"\{list_hist[img]}"
                    os.remove(hist)
    if 'wos' in sys.argv:
        folder_dir = his_path + rf"\wo spline"
        list_hist = os.listdir(folder_dir)
        for img in range(len(list_hist)):
            if 'jpg' in list_hist[img]:
                hist = folder_dir + rf"\{list_hist[img]}"
                os.remove(hist)
    if 'ws' in sys.argv:
        folder_dir = his_path + rf"\w spline"
        list_hist = os.listdir(folder_dir)
        for img in range(len(list_hist)):
            if 'jpg' in list_hist[img]:
                hist = folder_dir + rf"\{list_hist[img]}"
                os.remove(hist)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# delete_chro()                                                               #
# Deletes images from folder 'Chromosomes/images'                             #
# Also, deletes labels.txt from 'Chromosomes'                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def delete_chro():
    labels = rf"{cd}\Chromosomes\labels.txt"
    os.remove(labels)
    chro_folder = rf"{cd}\Chromosomes\images"
    chro_list = os.listdir(chro_folder)

    for c in range(len(chro_list)):
        if 'jpg' in chro_list[c]:
            chro = chro_folder + rf"\{chro_list[c]}"
            os.remove(chro)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# mosaic()                                                                    #
# Gets images from Mosaic/create and creates a mosaic                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def mosaic():
    mosaic_list = []
    save_dir = rf"{cd}\Mosaic\create"
    mosaic = []

    if 'j' not in sys.argv:
        if 'u' in sys.argv or 'b' in sys.argv:
            imgname = sys.argv[3]
        else:
            imgname = sys.argv[2]
        imgdir = rf"{cd}\Unmarked\{imgname}.jpg"
        img = cv2.imread(imgdir)
        _, black = cv2.threshold(img, 255, 255, cv2.THRESH_BINARY)
        if 'b' in sys.argv and not 'u' in sys.argv:  # Use black image as first image in the mosaic
            mosaic_list.append(black)
        elif 'u' in sys.argv:  # Use untreated image as first image in the mosaic
            mosaic_list.append(img)

        if 'u' in sys.argv or 'b' in sys.argv:
            nconfigs = len(sys.argv) - 4
        else:
            nconfigs = len(sys.argv) - 3

        for i in range(nconfigs):
            if 'b' in sys.argv or 'u' in sys.argv:
                config = sys.argv[i + 4].lower()
            else:
                config = sys.argv[i + 3].lower()
            config_dir = cd
            features = ""
            additions = True

            if 's' in config and not 'n' in config:  # Spline
                config_dir += rf"\Processed_spline"
            else:  # Otsu
                config_dir += rf"\Processed_otsu"
                additions = False

            if 'o' in config and not 'c' in config:  # Open
                config_dir += rf"\Open "
                features += 'O'
            else:  # Close
                config_dir += rf"\Close "
                features += 'C'

            if '2' in config:  # Kernel 2
                config_dir += "2"
                features += "2"
            elif '3' in config:  # Kernel 3
                config_dir += "3"
                features += "3"
            elif '4' in config:  # kernel 4
                config_dir += "4"
                features += "4"
            elif '5' in config:  # kernel 5
                config_dir += "5"
                features += "5"

            if additions:
                config_dir += rf"\Addition "
                if "aaa" in config:
                    config_dir += rf"30"
                    features += "a30"
                if "aa" in config:
                    config_dir += rf"20"
                    features += "a20"
                if "a" in config:
                    config_dir += rf"10"
                    features += "a10"
                else:
                    config_dir += rf"00"
                    features += "a00"

            config_dir += rf"\{imgname}_P({features}).jpg"
            img_c = cv2.imread(config_dir) # Read specified file
            mosaic_list.append(img_c)

            if 'mv' in sys.argv and not 'mh' in sys.argv:
                mosaic = np.vstack(mosaic_list)
            elif 'mh' in sys.argv and not 'mv' in sys.argv:
                mosaic = np.hstack(mosaic_list)



    else:
        nimg = len(sys.argv) - 3  # sys.argv -process.py -mv -j
        create_dir = rf"{cd}\Mosaic\create"
        jref = sys.argv.index('j') + 1


        for ind in range(nimg):
            print(create_dir + rf"\{sys.argv[jref + ind]}.jpg")
            mosaic_list.append(cv2.imread(create_dir + rf"\{sys.argv[jref + ind]}.jpg"))

        if 'mh' in sys.argv and not 'mv' in sys.argv:
            mosaic = np.hstack(mosaic_list)
        elif 'mv' in sys.argv and not 'mh' in sys.argv:
            mosaic = np.vstack(mosaic_list)

    mosaicname = f"mosaic{ len(os.listdir(save_dir))+1}.jpg"
    cv2.imwrite(mosaicname, mosaic)
    shutil.move(rf"{cd}\{mosaicname}", save_dir)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  Execution                                                                        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  Default Configuration                                                            #
# > process.py    : Processes images and stores them in Processed (method: spline)  #
#  process.py n   : Processes images and stores them in Processed (method: otsu)    #
# > process.py np : Forces non-execution of default configuration                   #
#                                                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                                   #
# > process.py c        : Cuts chromosomes, stores them in Chromosomes/images and   #
#                         creates labels.txt                                        #
# > process.py h        : Gets histograms plots (with spline curve, by default)     #
# > process.py h wos    : Gets histograms plots without spline curve                #
# > process.py h d      : Deletes images from Histograms/*                          #
# > process.py h d wos  : Deletes images from Histograms/wo spline                  #
# > process.py h d ws   : Deletes images from Histograms/w spline                   #
#                                                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                                   #
# > process.py d    : Deletes images from main directory                            #
# > process.py d s  : Deletes images from /Processed_spline/*                       #
# > process.py d n  : Deletes images from /Processed_otsu/*                         #
# > process.py c d  : Deletes images and label.txt from Chromosomes directory       #
#                                                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Mosaic                                                                            #
# > process.py mv ...    > process.py mh ...                                        #
#                                                                                   #
# [1] mh for horizontal mosaic, mv for vertical mosaic                              #
# [2] name of image whose config is to be compared (DO NOT ADD .jpg)                #
# [3...n] each desired configuration to appear in the mosaic                        #
#     * s for spline, n for otsu                                                    #
#     * o for open, c for close                                                     #
#     * 2 for 2x2, 3 for 3x3, 4 for 4x4, 5 for 5x5                                  #
#     * a for addition 10, aa for addition 20, aaa for addition 30                  #
#                                                                                   #
# Additional console args:                                                          #
#     * b to add black image to mosaic                                              #
#     * u to add untreated image to mosaic (image from Unmarked)                    #
#     * j to join previously created mosaic (only accepts names of                  #
#        images placed in Mosaic/create                                             #
#                                                                                   #
# Examples:                                                                         #
# > process.py mh u 2Gy-004 so2 so3 so4 so5                                         #
# > process.py mh b 2Gy-004 sc2 sc3 sc4 sc5                                         #
# > process.py mv j mosaic1 mosaic2                                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# > main.py
if 'd' not in sys.argv and 'np' not in sys.argv and not 'mv' in sys.argv and not 'mh' in sys.argv:
    process_images()      # TODO: Arreglar para Otsu!!
    images_to_processed() # TODO: Arreglar para Otsu!!

# > main.py c
if 'c' in sys.argv and 'd' not in sys.argv and not 'mv' in sys.argv and not 'mh' in sys.argv:
    chromosome_cut()
    images_to_chro()

# > main.py h
if 'h' in sys.argv and 'd' not in sys.argv:
    plot_hist()
    images_to_hist()

# > main.py d
if 'd' in sys.argv and 'h' not in sys.argv and not 'c' in sys.argv and not 's' in sys.argv and not 'n' in sys.argv:
    delete_images()

# > main.py d
if 'd' in sys.argv and ('n' in sys.argv or 's' in sys.argv):
    delete_proc()

# > main.py c d
if 'c' in sys.argv and 'd' in sys.argv and not 'mv' in sys.argv and not 'mh' in sys.argv:
    delete_chro()

# > main.py h d
if 'h' in sys.argv and 'd' in sys.argv:
    delete_hist()

# > main mv / main mh
if 'mv' in sys.argv or 'mh' in sys.argv:
    mosaic() #TODO: Cambiar nombre de archivo de guardado (con while)
