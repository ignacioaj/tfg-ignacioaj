> 📋 A template README.md for code accompanying a Machine Learning paper

# Dicentric Chromosome Detection

## Context
Dichentric chromosome counting is a technique widely used in biological dosimetry to quantify the genetic damage caused by exposure to radioactivity. Traditionally, this task has been performed manually by health professionals. However, the current interest is focused on achieving the automation of this process through Artificial Intelligence tools to achieve a reduction in time and an improvement in the quality of the results. This project aims to contribute to the state of the art of this study which, to date, there is still little research.

In this work, a detailed study will be carried out in which a convolutional neural network will be trained from a set of images on which different preprocessing techniques will be applied. The predictions produced by the model for each of the experiments will be analysed in order to determine which preprocessing technique provides the best predictions.

## Scope
This project represents the bachelor thesis work to complete my Degree in Biomedical Engineering at Univeristy of Málaga.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Project files

- ```setup.py```: Script for project setup and project secondary functionalities.
- ```process.py```: Script for image preprocessing.
- ```train.ipynb```: Script for model training and validation.
- ```ensemble.py```: Script for data ensembling from model predictions.
- ```Unmarked.zip```: Image set courtesy of the center REAC/TS of Institute ORISE from Tennessee, U.S.
- ```Marked.zip```: Image set with anotations (these annotations allowed image labelling)
- ```Labels.zip```: Label set indicating chromosome location and class.

## Setting up work environment
Create a folder where the whole project will be hosted. Then, download ```setup.py```, ```Unmarked.zip``` and ```Labels.zip``` and place them in that folder (from now on, project root directory). ```Marked.zip``` is not neccesary for the project, however, it can be also unzipped to have a glance over which of the chromosomes are actually dicentric chromosomes.

 Run ```setup.py``` to have all the neccesary folders for the project created.

## Preprocessing images
Download ```process.py``` and place it into project root folder. 
1. Run ```process.py``` to have all the images processed. Once done, you can:
   - Click on ```Run with Parameters``` and add ```h``` in ```Script Parameters``` to: get all the histograms (showing spline curve) for the Spline preprocessing.
   - Click on ```Run with Parameters``` and add ```h wos``` in ```Script Parameters``` to: get all the histograms (without showing spline curve) for the Spline preprocessing.
3. Open ```setup.py```, click on ```Run with Parameters``` and add ```k``` in ```Script Parameters```to: split dataset into train and validation sets (using 5-Fold Cross Validation technique) for every experiment (from just-created preprocessed images). There are additional parameters that can be set:

      - ```k c xxx```: to apply algorithm on single experiment ```xxx``` instead of applying algorithm to all experiments.
        > 📋 Let xxx be a 3 character config code, where first character stands for thresholding technique (s for spline, n for otsu),
        > second character stands for morph filter (o for open c for close) and third character stands for filter size (2 for 2x2, 3 for 3x3, 4 for 4x4 and 5 for 5x5).
        > 
        > Example: so2 stands for spline open 2x2.

      - ```k f xxx```: to set number of folds to ```xxx``` instead of 5.
      - ```k s xxx```: to set seed value to your desired number value ```xxx``` (seed that sets image order randomization)
      - ```k d```: to delete all files from ```KFold-Cross Validation/*``` 

## Training and Validating Model
1. Upload zips located in ```KFold-Cross Validation``` folder to Google Drive (it is recommended to place them in the folder ```/Colab Notebooks```). Remember the location you place them for next steps.
2. Download ```train.ipynb``` and open it in Google Colab.
3. Follow the instructions in the .ipynb and run cells one by one. Note that training process takes too much time (around 20 minutes each), that's why iterating all executions was avoided.
4. Upload generated ```runs``` folder to GDrive (following the instructions in the .ipynb) and download it from there.
5. Extract the downloaded zip and move the folder whose name is runs into Runs folder (from the project root directory). It is highly recommended to rename the experiment runs folder ```runs``` to ```Runs xxx```, being ```xxx``` the experiment config code, for example: ```Runs so2``` , ```Runs nc3``` , ```Runs no4``` , ...

## Ensembling
Download ```ensemble.py``` and place it into project root folder.
1. Open ```setup.py``` and click on ```Run with Parameters``` and add ```vl``` in ```Script Parameters``` to: merge all validation labels from all splits into a single folder (this is neccesary for the ensemble script to work).
2. Open and run ```process.py``` to generate the excel files corresponding to both phases (phase 1: detection and phase 2: classification) of the ensemble. Three kind of .xlsx will be generated for every image ```xxx``` in every phase:
   - ```xxx_detected.xlsx``` : Columns stand for experiments and rows stand for chromosomes in the image. Cells with ```1``` had its chromosome detected for its experiments, whereas cells with ```0``` did not.
   - ```xxx_bbox.xlsx``` : Columns stand for experiments and rows stand for chromosomes in the image. Every cell stores coordinates and class of the prediction of the cromosome if it was detected (```-``` if it was not).
   - ```xxx_confusion.xlsx``` : Columns stand for experiments and rows stand for prediction type. Every cell stores a count of its prediction type for its experiment.
3. Click on ```Run with Parameters``` and add ```m``` in ```Script Parameters``` to: generate the excel metrics files corresponding to both phases (phase 1: detection and phase 2: classification) of the ensemble.

## Additional ```setup.py``` Functionalities

- ```setup.py d``` : Deletes images from project root directory.                                 
- ```setup.py d s``` : Deletes images from ```/Processed_spline/*```.                            
- ```setup.py d n``` : Deletes images from ```/Processed_otsu/*```.

### Mosaic

### Paint
Once ```xxx_bbox.xlsx``` has been generated, an image showing the prediction on detected chromosomes can be generated. Ground truth bounding box will be marked in a thin black rectangle, whereas bounding box predictions for dicentric and non-dicentric will be marked with wider rectangles (red for dicentric, green for non-dicentric). It should be noted that detection false positives are NOT shown. 

```p xxx yyy ccc``` where xxx stands for the name of the image to be edited, yyy is the desired experiment to be studied and ccc the specific chromosome to be marked (optional, if this last parameter is not included, all chromosomes will be marked). Example: ```p 2Gy-023 nc4 5``` to paint prediction for chromosome of index 5 of image 2Gy-023.jpg (otsu close 4x4 preprocessing).

> 📋 Let xxx be a 3 character config code, where first character stands for thresholding technique (s for spline, n for otsu),
> second character stands for morph filter (o for open c for close) and third character stands for filter size (2 for 2x2, 3 for 3x3, 4 for 4x4 and 5 for 5x5).
>
> Example: so2 stands for spline open 2x2.



