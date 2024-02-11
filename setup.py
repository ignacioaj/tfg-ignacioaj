import sys
import os

import pandas as pd
import numpy as np
from sklearn import model_selection
import yaml
import shutil
import cv2
from PIL import Image

VERSION = '0.0.1'
DESCRIPTION = 'Trabajo Fin de Grado'
LONG_DESCRIPTION = 'Trabajo Fin de Grado centrado en la detección de cromosomas dicéntricos'

cd = os.getcwd()  # get current directory
dir_path = rf"{cd}\Unmarked"
answer_path = rf"{cd}\Marked"
labels_dir = rf"{cd}\Labels"
spline_dir = rf'{cd}\Processed_spline'
otsu_dir = rf'{cd}\Processed_otsu'
runs_dir = rf'{cd}\Runs'
hist_dir = rf'{cd}\Histograms'
ensemble_dir= rf"{cd}\Ensemble"
mosaic_dir = rf'{cd}\Mosaic'
kfold_dir = rf"{cd}\KFold-Cross Validation"


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# create_folders():                                                           #
# Creates all the folders of the project                                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def create_folders():
    folders = ["Ensemble\create", "Ensemble\Metrics\Phase 1", "Ensemble\Metrics\Phase 2", "Ensemble\Phase 1", "Ensemble\Phase 2", "Ensemble\create""Histograms\w spline", "Histograms\w spline", "Mosaic", "Chromosomes\images","KFold-Cross Validation"]
    processed = ["Open 2", "Open 3", "Open 4", "Open 5", "Close 2", "Close 3", "Close 4", "Close 5"]
    additions = ["Addition 00", "Addition 10", "Addition 20", "Addition 30"]

    for f in range(len(folders)):
        if not os.path.isdir(folders[f]):  # Create folders
            os.makedirs(folders[f])

    for exp in range(len(processed)):
        for a in range(len(additions)):
            if not os.path.isdir(rf'{spline_dir}\{processed[exp]}'):  # Create \Processed_spline\*
                os.makedirs(rf'{spline_dir}\{processed[exp]}')

    for exp in range(len(processed)):
        if not os.path.isdir(rf'{otsu_dir}\{processed[exp]}'):  # Create \Processed_otsu\*
            os.makedirs(rf'{otsu_dir}\{processed[exp]}')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# yaml-create():                                                              #
# Gets source of images+labels and creates YAML file                          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def yaml_create(x):
    cd = os.getcwd()
    classes = {
        '0': 'non-dic',
        '1': 'dic'
    }
    yaml_dir = rf'{cd}/data.yaml'
    if not os.path.isfile(yaml_dir):
        with open(yaml_dir, 'w') as y:
            yaml.safe_dump({
                'path': rf'/content/KFold-Cross Validation/Fold-{x}',
                'train': 'train',
                'val': 'val',
                'names': classes,
            }, y)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# config_search(config):                                                      #
# Gets configuration in a single-letter code and returns config-dir and       #
# features (feature code stated in processed images name).                    #
#                                                                             #
# Example:                                                                    #
# 'so2' --> config_search() -->                                               #
# --> ['{cd}\Processed_spline\Close 2\Addition 0', 'C2']                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def config_search(config):
    config_dir = cd
    features = ''
    additions = True
    if not config == '':
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
            elif "aa" in config:
                config_dir += rf"20"
                features += "a20"
            elif "a" in config:
                config_dir += rf"10"
                features += "a10"
            else:
                config_dir += rf"00"
                features += "a00"
    else:
        config_dir = rf"{cd}\Unmarked"

    return [config_dir, features]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# kfolding():                                                                 #
#                                                                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def kfolding(ksplit, seed, config):

    if not os.path.isdir(kfold_dir):
        os.makedirs(kfold_dir)
    label_dir = rf"{cd}\Labels"
    label_list = os.listdir(label_dir)
    kf = model_selection.KFold(n_splits=int(ksplit), shuffle=True,
                               random_state=seed)  # Creamos el objeto de KFolding con k=5 folds
    kfolds = list(kf.split(label_list))

    # Search dir of images specified in config
    [config_dir, features] = config_search(config)

    # Create dirs and copy images from source
    for ind_setfold in range(len(kfolds)):
        fold_dir = rf"{kfold_dir}\Fold-{ind_setfold + 1}"
        tr_img_dir = rf"{fold_dir}\train\images"
        tr_label_dir = rf"{fold_dir}\train\labels"
        val_img_dir = rf"{fold_dir}\val\images"
        val_label_dir = rf"{fold_dir}\val\labels"

        if not os.path.isdir(fold_dir):                          # If folders don't exist, create them
            os.makedirs(tr_img_dir)
            os.makedirs(tr_label_dir)
            os.makedirs(val_img_dir)
            os.makedirs(val_label_dir)

        for ind_fold in range(len(kfolds[ind_setfold][0])):      # Move images and labels to */train
            ind_img = kfolds[ind_setfold][0][ind_fold]

            lb_name = label_list[ind_img].split('.')[0]
            img_name = rf"{lb_name}_P({features}).jpg"

            shutil.copy(rf"{config_dir}\{img_name}", tr_img_dir)
            os.rename(rf"{tr_img_dir}\{img_name}", rf"{tr_img_dir}\{lb_name}.jpg")   # Image name must match with label name
            shutil.copy(rf"{cd}\Labels\{lb_name}.txt", tr_label_dir)

        for ind_fold in range(len(kfolds[ind_setfold][1])):                                   # Move images and labels to */val
            ind_img = kfolds[ind_setfold][1][ind_fold]

            lb_name = label_list[ind_img].split('.')[0]
            img_name = rf"{lb_name}_P({features}).jpg"

            shutil.copy(rf"{config_dir}\{img_name}", val_img_dir)
            os.rename(rf"{val_img_dir}\{img_name}",rf"{val_img_dir}\{lb_name}.jpg")  # Image name must match with label name
            shutil.copy(rf"{cd}\Labels\{lb_name}.txt", val_label_dir)

        if 'y' in sys.argv:
            yaml_create(ind_setfold+1)
            shutil.move(rf"{cd}\data.yaml", fold_dir)

    print(f'KFolded from {config_dir}')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# kfolds_delete():                                                            #
# Delete all files from KFold-Cross Validation/*                              #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def kfolds_delete():
    folds = os.listdir(rf"{cd}\KFold-Cross Validation")
    for f in range(len(folds)):                                    # *\Fold f
        sets_dir = rf"{cd}\KFold-Cross Validation\{folds[f]}"
        sets = os.listdir(sets_dir)
        if os.path.isfile(rf'{sets_dir}\data.yaml'):
            os.remove(rf'{sets_dir}\data.yaml')
        for s in range(len(sets)):                                 # *\Fold f\s (s=train,s=val)
            filestype_dir = rf"{sets_dir}\{sets[s]}"
            filestype = os.listdir(filestype_dir)
            for ff in range(len(filestype)):                       # *\Fold f\s\ff (ff=images, ff=labels)
                files_dir = rf"{filestype_dir}\{filestype[ff]}"
                file_list = os.listdir(files_dir)
                for fff in range(len(file_list)):                  # *\Fold f\s\ff\image_fff.jpg
                    os.remove(rf"{files_dir}\{file_list[fff]}")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# create_valabel():                                                           #
# Merge all validation labels from every different split into a unique        #
# folder. This applies to all the experiments.                                #
#                                                                             #
# Note: Model must be previously trained and results must be placed in        #
#       Runs/* for this method to work.                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def create_valabel():
    runs = os.listdir(runs_dir)
    for exp in range(len(runs)):
        exp_dir = rf'{runs_dir}\{runs[exp]}'
        valabel_dir = rf'{exp_dir}\Validation Labels'
        if not os.path.isdir(valabel_dir):
            os.makedirs(valabel_dir)
        exp_content = os.listdir(exp_dir)
        vals = []
        for f in range(len(exp_content)):
            if 'val' in exp_content[f]:
                vals.append(exp_content[f])
        for v in range(len(vals)):
            vlabels_dir = rf'{exp_dir}\val{v + 1}\labels'
            vlabels = os.listdir(vlabels_dir)
            for l in range(len(vlabels)):
                label = rf'{vlabels_dir}\{vlabels[l]}'
                shutil.copy(label, valabel_dir)



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
    mosaic = []

    if 'j' not in sys.argv:
        if 'u' in sys.argv or 'b' in sys.argv:
            imgname = sys.argv[3]
        else:
            imgname = sys.argv[2]
        imgdir = rf"{cd}\Unmarked\{imgname}.jpg"
        img = cv2.cvtColor(cv2.imread(imgdir), cv2.COLOR_BGR2GRAY)
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
                elif "aa" in config:
                    config_dir += rf"20"
                    features += "a20"
                elif "a" in config:
                    config_dir += rf"10"
                    features += "a10"
                else:
                    config_dir += rf"00"
                    features += "a00"

            config_dir += rf"\{imgname}_P({features}).jpg"

            img_c = cv2.cvtColor(cv2.imread(config_dir), cv2.COLOR_BGR2GRAY)  # Read specified file
            mosaic_list.append(img_c)

            if 'mv' in sys.argv and not 'mh' in sys.argv:
                mosaic = np.vstack(mosaic_list)
            elif 'mh' in sys.argv and not 'mv' in sys.argv:
                mosaic = np.hstack(mosaic_list)
    else:
        nimg = len(sys.argv) - 3  # sys.argv -process.py -mv -j
        create_dir = rf"{cd}\Mosaic"
        jref = sys.argv.index('j') + 1


        for ind in range(nimg):

            mosaic_list.append(cv2.imread(create_dir + rf"\{sys.argv[jref + ind]}.jpg"))

        if 'mh' in sys.argv and not 'mv' in sys.argv:
            mosaic = np.hstack(mosaic_list)
        elif 'mv' in sys.argv and not 'mh' in sys.argv:
            mosaic = np.vstack(mosaic_list)

    mnum = 1
    while os.path.isfile(rf'{mosaic_dir}\mosaic{mnum}.jpg'):
        mnum += 1
    cv2.imwrite(f"mosaic{mnum}.jpg", mosaic)
    shutil.move(rf"{cd}\mosaic{mnum}.jpg", mosaic_dir)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# paint_bb(image_name,exp,chro_sel)                                                               #
# Deletes images from folder 'Chromosomes/images'                             #
# Also, deletes labels.txt from 'Chromosomes'                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def paint_bb(image_name,exp,chro_sel):
    img = cv2.imread(rf'{dir_path}\{image_name}.jpg')
    df = pd.read_excel(rf'{ensemble_dir}\Phase 1\{image_name}_bbox.xlsx')

    if chro_sel == '*':
        iterable0 = 0
        iterable1 = len(df)
        feature = ''
    else:
        iterable0 = int(chro_sel)
        iterable1 = int(chro_sel)+1
        feature = f'_chro{chro_sel}'

    for chro in range(iterable0, iterable1):
        tcoords = df['Ground Truth'][int(chro)].split(' ')
        pcoords = df[f'Runs {exp}'][int(chro)]

        xt1 = int(tcoords[0]);
        xt2 = int(tcoords[1]);
        yt1 = int(tcoords[2]);
        yt2 = int(tcoords[3])

        tcolor = (0, 0, 0)        # ground truth bounding box color : black
        start = (xt1, yt1)
        end = (xt2, yt2)
        img = cv2.rectangle(img, start, end, tcolor, 1)

        if '-' not in pcoords:
            pcoords = pcoords.split(' ')
            xp1 = int(pcoords[0])
            xp2 = int(pcoords[1])
            yp1 = int(pcoords[2])
            yp2 = int(pcoords[3])
            pclass = int(pcoords[4])
            if pclass == 0:
                pcolor = (0, 255, 0)  # prediction bounding box color : green (Normal chromosome)
            else:
                pcolor = (0, 0, 255)  # prediction bounding box color : red   (Abnormal chromosome)
            start = (xp1, yp1)
            end = (xp2, yp2)
            img = cv2.rectangle(img, start, end, pcolor, 2)

    cv2.imwrite(image_name + f"_{exp}{feature}.jpg", img)
    if os.path.isfile(rf'{ensemble_dir}\create\{image_name}_{exp}{feature}.jpg'):
        os.remove(rf'{ensemble_dir}\create\{image_name}_{exp}{feature}.jpg')
    shutil.move(rf'{cd}\{image_name}_{exp}{feature}.jpg', rf'{ensemble_dir}\create')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  Execution                                                                            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# > setup.py        => Creates all the folders of the project                           #
# > setup.py k ...  => K-Fold Cross Validation                                          #
# > setup.py vl ... => Create Validation Labels folder in every Runs/* folder           #
# > setup.py k ...  => K-Fold Cross Validation                                          #
#                                                                                       #
#  - d=... to delete all files from KFold-Cross Validation/*                            #
#  - s=... to set seed value                                                            #
#  - f=... to set number of folds (5 by default)                                        #
#  - c=... to set configurations                                                        #
#     * s for spline, n for otsu                                                        #
#     * o for open, c for close                                                         #
#     * 2 for 2x2, 3 for 3x3, 4 for 4x4, 5 for 5x5                                      #
#     * a for addition 10, aa for addition 20, aaa for addition 30                      #
#                                                                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                                       #
# > setup.py d    => Deletes images from main directory                                 #
# > setup.py d s  => Deletes images from /Processed_spline/*                            #
# > setup.py d n  => Deletes images from /Processed_otsu/*                              #
# > setup.py c d  => Deletes images and label.txt from Chromosomes directory            #
#                                                                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Mosaic                                                                                #
# > process.py mv ...               > process.py m                                      #
#                                                                                       #
# [1] mh for horizontal mosaic, mv for vertical mosaic                                  #
# [2] name of image whose config is to be compared (DO NOT ADD .jpg)                    #
# [3...n] each desired configuration to appear in the mosaic                            #
#     * s for spline, n for otsu                                                        #
#     * o for open, c for close                                                         #
#     * 2 for 2x2, 3 for 3x3, 4 for 4x4, 5 for 5x5                                      #
#     * a for addition 10, aa for addition 20, aaa for addition 30                      #
#                                                                                       #
# Additional console args:                                                              #
#     * b to add black image to mosaic                                                  #
#     * u to add untreated image to mosaic (image from Unmarked)                        #
#     * j to join previously created mosaic (only accepts names of                      #
#        images placed in Mosaic/create                                                 #
#                                                                                       #
# Examples:                                                                             #
# > process.py mh u 2Gy-004 so2 so3 so4 so5                                             #
# > process.py mh b 2Gy-004 sc2 sc3 sc4 sc5                                             #
# > process.py mv j mosaic1 mosaic2                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Paint                                                                                 #
# > process.py p ...                                                                    #
# [1] p                                                                                 #
# [2] name of image to be painted (DO NOT ADD .jpg)                                     #
# [3] Desired experiment configuration to be painted                                    #
#     * s for spline, n for otsu                                                        #
#     * o for open, c for close                                                         #
#     * 2 for 2x2, 3 for 3x3, 4 for 4x4, 5 for 5x5                                      #
#     * a for addition 10, aa for addition 20, aaa for addition 30                      #
# [4] Index of the desired chromosome whose bbox is to be plotted, if any               #
#     Not including any (or adding '*' instead of index) will plot all bounding boxes   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# > setup.py
if len(sys.argv) == 1:
    create_folders()
else:
    # > setup.py k
    if 'k' in sys.argv and not 'd' in sys.argv:
        if 'd' in sys.argv:
            kfolds_delete()
        elif 'z' in sys.argv:
            if '*' in sys.argv:
                config = ['no2', 'no3', 'no4', 'no5', 'nc2', 'nc3', 'nc4', 'nc5',
                          'so2', 'so3', 'so4', 'so5', 'sc2', 'sc3', 'sc4', 'sc5']

                # TODO: zip config[i]
            # else:    TODO: zip config

        else:
            if 'c' in sys.argv:  # Set config
                ind = sys.argv.index('c') + 1
                config = sys.argv[ind]
            else:
                config = ''  # Default config

            if 's' in sys.argv:  # Set seed
                ind = sys.argv.index('s') + 1
                seed = sys.argv[ind]
            else:
                seed = round(np.log(2) * 100)  # Default seed

            if type(sys.argv[sys.argv.index('k') + 1]) == int:  # Set k-splits
                ind = sys.argv.index('k') + 1
                splits = sys.argv[ind]
            else:
                splits = 5  # Default k-splits

            if '*' in sys.argv:
                config = ['no2', 'no3', 'no4', 'no5', 'nc2', 'nc3', 'nc4', 'nc5',
                          'so2', 'so3', 'so4', 'so5', 'sc2', 'sc3', 'sc4', 'sc5']
                for i in range(len(config)):
                    kfolding(splits, seed, config[i])
            else:
                kfolding(splits, seed, config)

    # > setup.py vl
    if 'vl' in sys.argv:
        create_valabel()

    # > setup.py c
    if 'c' in sys.argv and 'd' not in sys.argv and not 'mv' in sys.argv and not 'mh' in sys.argv:
        chromosome_cut()
        images_to_chro()

    # > setup.py d
    if 'd' in sys.argv and 'h' not in sys.argv and not 'c' in sys.argv and not 's' in sys.argv and not 'n' in sys.argv:
        delete_images()

    # > setup.py d
    if 'd' in sys.argv and ('n' in sys.argv or 's' in sys.argv):
        delete_proc()

    # > setup.py c d
    if 'c' in sys.argv and 'd' in sys.argv and not 'mv' in sys.argv and not 'mh' in sys.argv:
        delete_chro()

    # > setup.py h d
    if 'h' in sys.argv and 'd' in sys.argv:
        delete_hist()

    # > setup mv / setup mh
    if 'mv' in sys.argv or 'mh' in sys.argv:
        mosaic()

    # > setup p
    if 'p' in sys.argv:
        if '*' in sys.argv or len(sys.argv) == 4:
            chro = '*'
        else:
            chro = sys.argv[4]

        paint_bb(sys.argv[2], sys.argv[3], chro)

