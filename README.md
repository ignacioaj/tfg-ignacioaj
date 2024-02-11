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

> 📋 Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Project files

- ```setup.py```:
- ```process.py```:
- ```setup.ipynb```:
- ```Unmarked.zip```: Image set courtesy of the center REAC/TS of Institute ORISE from Tennessee, U.S.
- ```Marked.zip```: Image set with anotations (these annotations allowed image labelling)
- ```Labels.zip```: Label set indicating chromosome location and class.

## Setup work environment
Create a folder where the whole project will be hosted. Then, download ```setup.py``` and place it in that folder (from now on, project root directory).
Run ```setup.py``` to have all the neccesary folders for the project created.
Then, unzip ```Unmarked.zip``` and ```Labels.zip``` into the project root directory. ```Marked.zip``` is not neccesary for the project, however, it can be also unzipped to have a glance over which of the chromosomes are actually dicentric chromosomes.

## Preprocess images
Download ```process.py``` and place it into project root folder. 
1. Run ```process.py``` to have all the images processed using Spline preprocessing. Once done, you can:
   - Click on ```Run with Parameters``` and add ```h``` in ```Script Parameters``` to: get all the histograms (showing spline curve) for the Spline preprocessing.
   - Click on ```Run with Parameters``` and add ```h wos``` in ```Script Parameters``` to: get all the histograms (without showing spline curve) for the Spline preprocessing. 
2. Click on ```Run with Parameters``` and add ```n``` in ```Script Parameters``` to: have all the images processed using Otsu preprocessing.
3. Open ```setup.py```, click on ```Run with Parameters``` and add ```k``` in ```Script Parameters```to: split dataset into train and validation sets (using K-Fold Cross Validation technique) for every experiment (from just-created preprocessed images).
      - ```k d```: to delete all files from ```KFold-Cross Validation/*``` 
      - ```k s xxx```: to set seed value to your desired number value ```xxx``` (seed that sets image order randomization)

## Training and Validation
1. Upload zips located in ```KFold-Cross Validation``` folder to Google Drive (it is recommended to place them in the folder ```/Colab Notebooks```). Remember the location you place them for next steps.
2. Download ```train.ipynb``` and open it in Google Colab.
3. Follow the instructions of the .ipynb and run cells one by one. Note that training process takes too much time (around 20 minutes each), that's why iterating all executions was avoided.
4. Upload generated ```runs``` folder to GDrive (following the instructions in the .ipynb) and download it from there.
5. Extract the downloaded zip and move the folder whose name is runs into Runs folder (from the project root directory). It is highly recommended to rename the experiment runs folder ```runs``` to ```Runs xxx```, being ```xxx``` the experiment config code, for example: ```Runs so2``` , ```Runs nc3``` , ```Runs no4``` , ...

## Ensemble
To evaluate my model on ImageNet, run:

## Additional ```setup.py``` Functionalities


```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

> 📋 Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z.

> 📋 Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable). Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name       | Top 1 Accuracy | Top 5 Accuracy |
| ---------------- | -------------- | -------------- |
| My awesome model | 85%            | 95%            |

> 📋 Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it.

## Contributing

> 📋 Pick a licence and describe how to contribute to your code repository.
