# Dicentric Chromosome Detection

## Scope
This project represents the bachelor thesis work to complete the Degree in Biomedical Engineering at Univerisity of Málaga.

## Context
Dichentric chromosome counting is a technique widely used in biological dosimetry to quantify the genetic damage caused by exposure to radioactivity. Traditionally, this task has been performed manually by health professionals. However, the current interest is focused on achieving the automation of this process through Artificial Intelligence tools to achieve a reduction in time and an improvement in the quality of the results. This project aims to contribute to the state of the art of this study which, to date, there is still little research.

In this work, a detailed study will be carried out in which a convolutional neural network will be trained from a set of images on which different preprocessing techniques will be applied. The predictions produced by the model for each of the experiments will be analysed in order to determine which preprocessing technique provides the best predictions.

## Project files

- ```setup.py```: Script for project setup and project secondary functionalities.
- ```process.py```: Script for image preprocessing.
- ```train.ipynb```: Script for model training and validation.
- ```ensemble.py```: Script for data ensembling from model predictions.
- ```Unmarked.zip```: Image set courtesy of the REAC/TS center of ORISE Institute from Tennessee, U.S.
- ```Marked.zip```: Image set with anotations (these annotations allowed image labelling)
- ```Labels.zip```: Label set indicating chromosome location and class.
- ```requirements.txt```: .txt file that contains all project module dependencies to be installed.

## Installing requirements
Open the ```cmd``` and reach the folder where the project will be hosted by using the ```cd``` command.
Once the folder has been reached, install the requirements by typing:
```setup
pip install -r requirements.txt
```

## Setting up work environment
Create a folder where the whole project will be hosted. Then, download ```setup.py```, ```Unmarked.zip``` and ```Labels.zip``` and place them in that folder (from now on, project root directory). ```Marked.zip``` is not neccesary for the project, however, it can be also unzipped to have a glance over which of the chromosomes are actually dicentric chromosomes.

 Run ```setup.py``` to have all the neccesary folders for the project created.

## Preprocessing images
Download ```process.py``` and place it into project root folder. 
1. Run ```process.py``` to have all the images processed. Once done, you can:
   - Click on ```Run with Parameters``` and add ```h``` in ```Script Parameters``` to: get all the histograms (showing spline curve) for the Spline preprocessing.
   - Click on ```Run with Parameters``` and add ```h wos``` in ```Script Parameters``` to: get all the histograms (without showing spline curve) for the Spline preprocessing.
2. Open ```setup.py```, click on ```Run with Parameters``` and add ```k``` in ```Script Parameters```to: split dataset into train and validation sets (using 5-Fold Cross Validation technique) for every experiment (from just-created preprocessed images). Additionally, there are other parameters that can be set:

      - ```k c xxx```: to apply algorithm on one single experiment ```xxx``` instead of applying algorithm to all experiments.
        > 📋 Let xxx be a 3 character config code, where first character stands for thresholding technique (s for spline, n for otsu),
        > second character stands for morph filter (o for open c for close) and third character stands for filter size (2 for 2x2, 3 for 3x3, 4 for 4x4 and 5 for 5x5).
        > 
        > Example: so2 stands for spline open 2x2.

      - ```k f xxx```: to set number of folds to ```xxx``` instead of 5.
      - ```k s xxx```: to set seed value to your desired number value ```xxx``` (seed that sets image order randomization)
      - ```k d```: to delete all files from ```KFold-Cross Validation/*```
        
A ready-to-train zip per experiment will be generated in ```KFold-Cross Validation``` folder (in project root directory). 

## Training and Validating Model
1. Upload zips located in ```KFold-Cross Validation``` folder to Google Drive (it is recommended to place them in the folder ```/Colab Notebooks```). Remember the location you place them for next steps.
2. Download ```train.ipynb``` and open it in Google Colab.
3. Follow the instructions in the .ipynb and run cells one by one. Note that training process takes too much time (around 20 minutes each), that's why iterating all executions was avoided.
4. Upload generated ```runs``` folder to GDrive (following the instructions in the .ipynb) and download it from there.
5. Extract the downloaded zip and move the folder whose name is ```detect``` into ```Runs``` folder (in the project root directory). It is highly recommended to rename the experiment runs folder ```detect``` to ```Runs xxx```, being ```xxx``` the experiment config code, for example: ```Runs so2``` , ```Runs nc3``` , ```Runs no4``` , ...

## Ensembling
Download ```ensemble.py``` and place it into project root folder.
1. Open ```setup.py``` and click on ```Run with Parameters``` and add ```vl``` in ```Script Parameters``` to: merge all validation labels from all splits into a single folder (this is neccesary for the ensemble script to work).
2. Open and run ```process.py``` to generate the excel files corresponding to both phases (phase 1: detection and phase 2: classification) of the ensemble. Three kind of .xlsx will be generated for every image ```xxx``` in every phase:
   - ```xxx_detected.xlsx``` : Columns stand for experiments and rows stand for chromosomes in the image. Cells with ```1``` had its chromosome detected for its experiments, whereas cells with ```0``` did not.
   - ```xxx_bbox.xlsx``` : Columns stand for experiments and rows stand for chromosomes in the image. Every cell stores coordinates and class of the prediction of the cromosome if it was detected (```-``` if it was not).
   - ```xxx_confusion.xlsx``` : Columns stand for experiments and rows stand for prediction type. Every cell stores a count of its prediction type for its experiment.
3. Click on ```Run with Parameters``` and add ```m``` in ```Script Parameters``` to: generate the excel metrics files corresponding to both phases (phase 1: detection and phase 2: classification) of the ensemble.

## Additional ```setup.py``` Functionalities
Further functionalities are provided by ```setup.py```. These can be used by clicking on ```Run with Parameters``` and adding the following parameters in ```Script Parameters``` :

### Delete images
- ```d``` : to delete images from project root directory.                                 
- ```d s``` : to delete images from ```/Processed_spline/*```.                            
- ```d n``` : to delete images from ```/Processed_otsu/*```.

### Mosaic
A mosaic of images can be generated in order to compare them. Use ```mh``` for horizontal mosaics (images height must be the same) and ```mv``` for vertical mosaics (images width must be the same). Generated mosaics will be saved in ```Mosaic``` folder of project root directory. 

- Compare different experiments for a single image: Use ```mh xxx yyy yyy yyy``` or ```mv xxx yyy yyy yyy``` (add as many ```yyy``` as desired), where ```xxx``` stands for the name of the image to be edited (excluiding '.jpg') and ```yyy``` are all the desired experiments to be studied. Additionally, you can use parameters ```u``` to add untreated image and ```b``` to add black image.

  Examples: ```mh u 2Gy-004 so2 so3 so4 so5``` , ```mh b 2Gy-004 sc2 sc3 sc4 sc5```
  > 📋 Let yyy be a 3 character config code, where first character stands for thresholding technique (s for spline, n for otsu),
  > second character stands for morph filter (o for open c for close) and third character stands for filter size (2 for 2x2, 3 for 3x3, 4 for 4x4 and 5 for 5x5).
  >
  > Example: so2 stands for spline open 2x2.

- Compare other images (thought for creating mosaics out of mosaics): Parameter ```j``` will compose a mosaic from images in ```Mosaics``` folder instead from the experiments. Use ```mh j xxx xxx xxx``` or ```mv j xxx xxx xxx``` (add as many ```xxx``` as desired) where ```xxx```  stands for the name of the image to be edited (excluiding '.jpg'). Example: ```mv j mosaic1 mosaic2```

### Paint
Once ```xxx_bbox.xlsx``` has been generated, an image showing the prediction on detected chromosomes can be generated. Ground truth bounding box will be marked in a thin black rectangle, whereas bounding box predictions for dicentric and non-dicentric will be marked with wider rectangles (red for dicentric, green for non-dicentric). It should be noted that detection false positives are NOT shown. 

Use ```p xxx yyy ccc``` , where ```xxx``` stands for the name of the image to be edited (excluiding '.jpg'), ```yyy``` is the desired experiment to be studied and ```ccc``` the specific chromosome to be marked (optional, if this last parameter is not included, all chromosomes will be marked).  Generated images will be saved in ```Ensemble/create``` folder of project root directory.

Example: ```p 2Gy-023 nc4 5``` to paint prediction for chromosome of index 5 of image 2Gy-023.jpg (otsu close 4x4 preprocessing).

> 📋 Let yyy be a 3 character config code, where first character stands for thresholding technique (s for spline, n for otsu),
> second character stands for morph filter (o for open c for close) and third character stands for filter size (2 for 2x2, 3 for 3x3, 4 for 4x4 and 5 for 5x5).
>
> Example: so2 stands for spline open 2x2.



