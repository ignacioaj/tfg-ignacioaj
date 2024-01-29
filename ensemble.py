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
ensemble_dir= rf"{cd}\Ensemble"


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# abs_coords():                                                               #
# Returns absolute coordinates x1,x2,y1,y2 given coordinates xrc,yrc,rw,rh    #
# relative to image width and height                                          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def abs_coords(xrc,yrc,rw,rh,width,height):
    xrc=float(xrc)             # x of the center of the bbox relative to image width
    yrc = float(yrc)           # y of the center of the bbox relative to image height
    rw = float(rw)             # bbox width relative to image width
    rh = float(rh)             # bbox height relative to image height

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
# ensemble()                                                              #
# For every image, returns a txt with a matrix stating which chromosomes      #
# were correctly predicted for every experiments. To this aim, ground truth   #
# chromosomes are compared to every predicted chromosome. If any of the       #
# predicted achieves iou>0.5 and same class, returns 1, else returns 0        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def ensemble():
    gd_tp = list(np.zeros(len(runs)))  # This matrix will store the number of total tp detected in phase 1
    gd_fp = list(np.zeros(len(runs)))  # This matrix will store the number of total fp detected in phase 1
    gd_fn = list(np.zeros(len(runs)))  # This matrix will store the number of total fn detected in phase 1
    gc_tp = list(np.zeros(len(runs)))  # This matrix will store the number of total tp detected in phase 2
    gc_tn = list(np.zeros(len(runs)))  # This matrix will store the number of total tn detected in phase 2
    gc_fp = list(np.zeros(len(runs)))  # This matrix will store the number of total fp detected in phase 2
    gc_fn = list(np.zeros(len(runs)))  # This matrix will store the number of total fn detected in phase 2

    for lt in range(len(labels)):                        # For every label in /Labels (which are ground truth labels)...

        d_tp = list(np.zeros(len(runs)))  # This matrix will store the number of tp detected in phase 1 (current image)
        d_fp = list(np.zeros(len(runs)))  # This matrix will store the number of fp detected in phase 1 (current image)
        d_fn = list(np.zeros(len(runs)))  # This matrix will store the number of fn detected in phase 1 (current image)
        c_tp = list(np.zeros(len(runs)))  # This matrix will store the number of tp detected in phase 2 (current image)
        c_tn = list(np.zeros(len(runs)))  # This matrix will store the number of tn detected in phase 2 (current image)
        c_fp = list(np.zeros(len(runs)))  # This matrix will store the number of fp detected in phase 2 (current image)
        c_fn = list(np.zeros(len(runs)))  # This matrix will store the number of fn detected in phase 2 (current image)

        with open(rf'{labels_dir}\{labels[lt]}','r') as label:
            lines = label.readlines()

            if not os.path.isdir(ensemble_dir):
                os.makedirs(ensemble_dir)

            p1_matrix = []   # This matrix will store which of the ground truth chromosomes were detected
            bb_matrix = []   # This matrix will store coordinates and class of bbox of predictions
            gt_column = []   # This column will store coordinates and class of bbox of ground truths

            for ct in lines:                              # ...Go through every ground truth chromosome
                ct_info = ct.split(' ')
                img = cv2.imread(rf"{cd}\Unmarked\{labels[lt].split('.')[0]}.jpg")
                p1_new_line = []
                bb_new_line = []

                height = img.shape[0]
                width = img.shape[1]
                tclass = ct_info[0]
                [xt1,xt2,yt1,yt2] = abs_coords(ct_info[1],ct_info[2],ct_info[3],ct_info[4],width,height)
                gt_column.append(rf'{xt1} {xt2} {yt1} {yt2} {tclass}')

                for exp in range(len(runs)):              # And for every experiment...

                    valabels_dir = rf'{cd}\Runs\{runs[exp]}\Validation Labels'
                    output = 0

                    with open(rf'{valabels_dir}\{labels[lt]}', 'r') as label_p:
                        lines_p = label_p.readlines()
                        cp = 0

                    max_iou = 0
                    best_coords = [0, 0, 0, 0]
                    best_class = -1
                    for cp in lines_p:     # ...ground truth is compared with every predicted chromosome
                        cp_info = cp.split(' ')
                        pclass = cp_info[0]
                        [xp1,xp2,yp1,yp2] = abs_coords(cp_info[1],cp_info[2],cp_info[3],cp_info[4],width,height)

                        if is_intersection(xt1,xt2,yt1,yt2,xp1,xp2,yp1,yp2):
                            at = get_area(xt1,xt2,yt1,yt2)
                            ap = get_area(xp1,xp2,yp1,yp2)
                            [xi1,xi2,yi1,yi2] = intersection_coords(xt1,xt2,yt1,yt2,xp1,xp2,yp1,yp2)
                            ai = get_area(xi1,xi2,yi1,yi2)                                    # Intersection area
                            au = at + ap - ai                                                 # Union area
                            iou = ai / au                                                     # Intersection over Union

                            if iou > 0.5 and iou > max_iou:
                                output = 1
                                max_iou = iou
                                best_coords = [xp1, xp2, yp1, yp2]
                                best_class = pclass

                    if output == 1:                                 # ground truth chromosome was detected => phase 1 TP
                        d_tp[exp] += 1
                        gd_tp[exp] += 1
                        bb_new_line.append(rf'{best_coords[0]} {best_coords[1]} {best_coords[2]} {best_coords[3]} {best_class}')

                        if tclass == best_class == '1':             # Detected and classes match (1-1) => phase 2 TP
                            c_tp[exp] += 1
                            gc_tp[exp] += 1
                        elif tclass == best_class == '0':           # Detected and classes match (0-0) => phase 2 TN
                            c_tn[exp] += 1
                            gc_tn[exp] += 1
                        elif tclass == '0' and best_class == '1':   # Detected and classes do not match (0-1) => phase 2 FP
                            c_fp[exp] += 1
                            gc_fp[exp] += 1
                        elif best_class == '0' and tclass == '1':   # Detected and classes do not match (1-0) => phase 2 FN
                            c_fn[exp] += 1
                            gc_fn[exp] += 1
                    else:                                           # ground truth chromosome was NOT detected => phase 1 FN
                        d_fn[exp] += 1
                        gd_fn[exp] += 1
                        bb_new_line.append(rf'-')

                    p1_new_line.append(output)


                p1_matrix.append(p1_new_line)
                bb_matrix.append(bb_new_line)


        p1_df = pd.DataFrame(data=p1_matrix, columns=runs)
        p1_df.to_excel(rf'{cd}\Ensemble\Phase 1\{labels[lt].split(".")[0]}_detected.xlsx')
        bb_df = pd.DataFrame(data=bb_matrix, columns=runs)
        bb_df.insert(0, "Ground Truth", gt_column, True)
        bb_df.to_excel(rf'{cd}\Ensemble\Phase 1\{labels[lt].split(".")[0]}_bbox.xlsx')
        for exp in range(len(runs)):
            valabels_dir = rf'{cd}\Runs\{runs[exp]}\Validation Labels'
            with open(rf'{valabels_dir}\{labels[lt]}', 'r') as label_p:
                predictions = len(label_p.readlines())
            d_fp[exp] = predictions - d_tp[exp]    # prediction does NOT match any ground truth chromosome => phase 1 FP
            gd_fp[exp] += d_fp[exp]

        d_df = pd.DataFrame(data=[d_tp, d_fp, d_fn],index=['TP', 'FP', 'FN'], columns=runs)
        d_df.to_excel(rf'{cd}\Ensemble\Phase 1\{labels[lt].split(".")[0]}_confusion.xlsx')
        c_df = pd.DataFrame(data=[c_tp, c_tn, c_fp, c_fn],index=['TP', 'TN', 'FP', 'FN'],columns=runs)
        c_df.to_excel(rf'{cd}\Ensemble\Phase 2\{labels[lt].split(".")[0]}_confusion.xlsx')

    d_df = pd.DataFrame(data=[gd_tp, gd_fp, gd_fn], index=['TP', 'FP', 'FN'], columns=runs)
    d_df.to_excel(rf'{cd}\Ensemble\Phase 1\general_confusion.xlsx')
    c_df = pd.DataFrame(data=[gc_tp, gc_tn, gc_fp, gc_fn], index=['TP', 'TN', 'FP', 'FN'], columns=runs)
    c_df.to_excel(rf'{cd}\Ensemble\Phase 2\general_confusion.xlsx')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# metrics()                                                                   #
# Given the xlsx tables from the previous ensemble function, metrics for      #
# every experiment of every image are returned                                #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def metrics():
    confusion_tables = os.listdir(rf'{ensemble_dir}\Phase 2')

    # Concrete image - Metrics
    for ctb in range(len(confusion_tables)):
        dmatrix = [] # Concrete image - detection metrics matrix
        cmatrix = [] # Concrete image - classification metrics matrix

        d_df = pd.read_excel(rf'{ensemble_dir}\Phase 1\{confusion_tables[ctb]}')
        c_df = pd.read_excel(rf'{ensemble_dir}\Phase 2\{confusion_tables[ctb]}')

        for exp in range(len(runs)):

            tp_d = d_df[f'{runs[exp]}'][0]
            fp_d = d_df[f'{runs[exp]}'][1]
            fn_d = d_df[f'{runs[exp]}'][2]
            tp_c = c_df[f'{runs[exp]}'][0]
            fp_c = c_df[f'{runs[exp]}'][2]
            fn_c = c_df[f'{runs[exp]}'][3]
            tn_c = c_df[f'{runs[exp]}'][1]

            if (tp_d + fn_d) != 0:
                rc_d  = tp_d / (tp_d + fn_d)
                fnr_d = fn_d / (tp_d + fn_d)
            else:
                rc_d  = 0
                fnr_d = 0
            if (tp_d + fp_d) != 0:
                pr_d = tp_d / (tp_d + fp_d)
            else:
                pr_d = 0
            if (pr_d + rc_d) != 0:
                fm_d = 2 * pr_d * rc_d / (pr_d + rc_d)
            else:
                fm_d = 0
            if (tp_d + fn_d + fp_d) != 0:
                s_d = tp_d / (tp_d + fn_d + fp_d)
            else:
                s_d = 0

            if (tp_c + fn_c) != 0:
                rc_c = tp_c / (tp_c + fn_c)
                fnr_c = fn_c / (tp_c + fn_c)
            else:
                rc_c = 0
                fnr_c = 0
            if (tp_c + tn_c) != 0:
                sp_c = tn_c / (tp_c + tn_c)
            else:
                sp_c = 0
            if (fp_c + tn_c) != 0:
                fpr_c = fp_c / (fp_c + tn_c)
            else:
                fpr_c = 0
            if (tp_c + fp_c) != 0:
                pr_c = tp_c / (tp_c + fp_c)
            else:
                pr_c = 0
            if (tp_c + fp_c + tn_c + fn_c) != 0:
                pwc_c = 100 * (fn_c + fp_c) / (tp_c + fp_c + tn_c + fn_c)
                acc_c = (tp_c + tn_c) / (tp_c + tn_c + fp_c + fn_c);
            else:
                pwc_c = 0
                acc_c = 0
            if(pr_c + rc_c) != 0:
                fm_c = (2 * pr_c * rc_c) / (pr_c + rc_c)
            else:
                fm_c = 0
            if(tp_c + fn_c + fp_c) != 0:
                s_c = tp_c / (tp_c + fn_c + fp_c)
            else:
                s_c = 0

            dmetrics = [rc_d, fnr_d, pr_d, fm_d, s_d]
            dmatrix.append(dmetrics)
            cmetrics = [rc_c, sp_c, fpr_c, fnr_c, pwc_c, pr_c, fm_c, s_c, acc_c]
            cmatrix.append(cmetrics)

        dmetrics_df = pd.DataFrame(data=np.array(dmatrix).transpose(),index=['RC','FNR','PR','FM','S'], columns=runs)
        dmetrics_df.to_excel(rf'{cd}\Ensemble\Metrics\Phase 1\{confusion_tables[ctb].split("_")[0]}_metrics.xlsx')

        cmetrics_df = pd.DataFrame(data=np.array(cmatrix).transpose(),index=['RC', 'SP', 'FPR', 'FNR', 'PWC', 'PR','FM','S','ACC'], columns=runs)
        cmetrics_df.to_excel(rf'{cd}\Ensemble\Metrics\Phase 2\{confusion_tables[ctb].split("_")[0]}_metrics.xlsx')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  Execution                                                                            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  > ensemble.py         : generate ensembles (format .xlsx)                            #
#  > ensemble.py m       : generate metrics (format .xlsx)                              #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if len(sys.argv) == 1:
    ensemble()
elif 'm' in sys.argv:
    metrics()
