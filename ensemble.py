import os
import sys
import cv2
import numpy as np
import pandas as pd

# Global variables
cd = os.getcwd()  # get current directory
dir_path = rf"{cd}\Unmarked"
answer_path = rf"{cd}\Marked"
labels_dir = rf"{cd}\Labels"
labels = os.listdir(labels_dir)
runs_dir = rf'{cd}\Runs'
runs = os.listdir(runs_dir)
ensemble_labels_dir= rf"{cd}\Ensemble\labels"

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# abs_coords():                                                               #
# Returns absolute coordinates x1,x2,y1,y2 given coordinates xrc,yrc,rw,rh    #
# relative to image width and height                                          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def abs_coords(xrc,yrc,rw,rh,width,height):
    xrc=float(xrc)
    yrc = float(yrc)
    rw = float(rw)
    rh = float(rh)

    xc = width * xrc           # absolute x of the center of the bbox
    yc = height * yrc          # absolute y of the center of the bbox
    wb = rw * width            # absolute width of the bbox
    hb = rh * height           # absolute height of the bbox

    x1 = round(xc - wb / 2)
    x2 = round(xc + wb / 2)
    y1 = round(yc - hb / 2)
    y2 = round(yc + hb / 2)

    return [x1,x2,y1,y2]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# is_intersection(xa1, xa2, ya1, ya2, xb1, xb2, yb1, yb2):                    #
# Given the coordinates x1,x2,y1,y2 of two different rectangles, returns      #
# true if the rectangles are coincident and false if the rectangles are not.  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def is_intersection(xa1,xa2,ya1,ya2,xb1, xb2, yb1, yb2):
    if (ya1 > yb2) or (yb1 > ya2) or (xa1 > xb2) or (xb1 > xa2):
        return False
    else:
        return True


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# get_area():                                                                 #
# Returns area of a rectangle given parameters x1,x2,y1,y2                    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def get_area(x1,x2,y1,y2):
    return (x2-x1)*(y2-y1)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# intersection_coords():                                                      #
# Returns coordinates x1,x2,y1,y2 of the interesection between 2 bounding     #
# boxes (bb) given their respective coordinates x1,x2,y1,y2                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def intersection_coords(xa1, xa2, ya1, ya2, xb1, xb2, yb1, yb2):
    xi1= max(xa1,xb1)
    xi2= min(xa2,xb2)
    yi1= max(ya1,yb1)
    yi2= min(ya2,yb2)
    return [xi1,xi2,yi1,yi2]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ensemble_txt    ()                                                          #
# For every image, returns a txt with a matrix stating which chromosomes      #
# were correctly predicted for every experiments. To this aim, ground truth   #
# chromosomes are compared to every predicted chromosome. If any of the       #
# predicted achieves iou>0.5 and same class, returns 1, else returns 0        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def ensemble_txt():

    for lt in range(len(labels)):            # For every label in /Labels (which are ground truth labels)

        with open(rf'{labels_dir}\{labels[lt]}','r') as label:
            lines = label.readlines()

            if not os.path.isdir(ensemble_labels_dir):
                os.makedirs(ensemble_labels_dir)

            matrix = ''

            for ct in lines:       # Goes through every ground truth chromosome
                ct_info = ct.split(' ')
                img = cv2.imread(rf"{cd}\Unmarked\{labels[lt].split('.')[0]}.jpg")

                height = img.shape[0]
                width = img.shape[1]
                tclass = ct_info[0]
                [xt1,xt2,yt1,yt2] = abs_coords(ct_info[1],ct_info[2],ct_info[3],ct_info[4],width,height)


                for exp in range(len(runs)):   # And for every experiment...

                    valabels_dir = rf'{cd}\Runs\{runs[exp]}\Validation Labels'

                    with open(rf'{valabels_dir}\{labels[lt]}', 'r') as label_p:
                        lines_p = label_p.readlines()
                        cp = 0

                    output = 0
                    
                    while (output==0) and (cp<=len(lines_p)-1):     # Ground truth is compared with every predicted chromosome
                        cp_info = lines_p[cp].split(' ')
                        pclass = cp_info[0]
                        [xp1,xp2,yp1,yp2] = abs_coords(cp_info[1],cp_info[2],cp_info[3],cp_info[4],width,height)

                        if is_intersection(xt1,xt2,yt1,yt2,xp1,xp2,yp1,yp2):
                            at = get_area(xt1,xt2,yt1,yt2)
                            ap = get_area(xp1,xp2,yp1,yp2)
                            [xi1,xi2,yi1,yi2] = intersection_coords(xt1,xt2,yt1,yt2,xp1,xp2,yp1,yp2)
                            ai = get_area(xi1,xi2,yi1,yi2)                                                # Intersection area
                            au = at + ap - ai                                                             # Union area
                            iou = ai / au                                                                 # Intersection over Union

                            if (iou > 0.5) and (tclass == pclass):
                                output = 1

                        cp+=1

                    if ct == len(lines) - 1:
                        matrix += f'{output}'
                    elif exp == len(runs)-1:
                        matrix += f'{output}\n'
                    else:
                        matrix += f'{output} '

                with open(rf'{ensemble_labels_dir}\{labels[lt]}', 'w') as new_label:
                    new_label.write(matrix)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ensemble_df()                                                               #
# Returns a Excell (.xlsx) of the ensemble labels once generated              #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def ensemble_df():
    ensemble_content = os.listdir(ensemble_labels_dir)
    for l in range(len(ensemble_content)):
        df = pd.read_csv(rf"{ensemble_labels_dir}\{ensemble_content[l]}", sep=" ", names=runs)
        df.to_excel(rf'{ensemble_labels_dir}\{ensemble_content[l].split(".")[0]}.xlsx')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ensemble_info()                                                               #
# Given the coordinates x1,x2,y1,y2 of two different rectangles, returns      #
# true if the rectangles are coincident and false if the rectangles are not.  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def ensemble_info():
    ensemble_content = os.listdir(ensemble_labels_dir)
    labels_txt = []
    matrix = []

    for l in range(len(ensemble_content)):
        if '.txt' in ensemble_content[l]:
            labels_txt.append(ensemble_content[l])
    for l in range(len(labels_txt)):
        with open (rf'{ensemble_labels_dir}\{labels_txt[l]}') as label:
            lines = label.readlines()
            sum = list(np.zeros(len(runs)))
            for line in lines:
                for exp in range(len(line.split(' '))):
                    sum[exp] += int(line.split(' ')[exp]) / len(lines) * 100
            matrix.append(sum)
    df = pd.DataFrame(data=matrix, index=labels_txt, columns=runs)
    df.to_excel(rf'{ensemble_labels_dir}\ensemble.xlsx')





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  Execution                                                                            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  > ensemble.py         : generate ensembles (format .txt)                             #
#  > ensemble.py df      : generate ensembles (formats .txt and .xlsx)                  #
#  > ensemble.py !txt df : generate ensembles (format .xlsx only!)                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if not '!txt' in sys.argv:
    ensemble_txt()

if 'df' in sys.argv:
    ensemble_df()

if 'i' in sys.argv:
    ensemble_info()