# smartnoisefilter: NoIze

<a href='https://aislynrose.bitbucket.io/'>Documentation</a>

<a href='https://notebooks.ai/a-n-rose/noize-demo-360a3df2'>Demo</a>

This smart noise filter was built under the project \\NoIze//, managed by Peggy Sylopp (contact@peggy-sylopp.net).

Author of the code/software in this repository: Aislyn Rose (rose.aislyn.noelle@gmail.com, <a href='https://a-n-rose.github.io/'>a-n-rose.github.io</a>)

## About

This software has four main parts: 1) pathway formation, 2) feature extraction, 3) scene classifier training, and 4) noise filtering. Below covers each section briefly.

### Pathway Formation

This step checks pathways and files. If necessary pathways don't exist yet, they are created. These pathways are created using the project name provided as well as the type of features indicated for extraction. This is done to organize newly created files in a way to avoid conflicts and overwriting data files.

### Feature Extraction

There are two kinds of features extracted:

1) features geared for model training

2) features geared for noise filtering

For both types of feature extraction, the features are orgnized accroding to their class. It expects all audiofiles belonging to one class be in the same folder. The name of that folder is used as a label for that class.

Model Training Features:

The features for the model can be either 'mfcc' or 'fbank' features. These are the features extracted from each audio file in the training dataset. These features are saved in .npy files (numpy files) to be later fed to the model.

Noise Filter Features:

The features for the filter are power spectrum averages. These are power averages collected from all audio files belonging to a single audio class. These power averages are used to inform the filter the appx. power spectrum of the noise it needs to reduce. 

### Scene Classifier Training

The model aka scene classifier is trained on the 'mfcc' or 'fbank' features previously extracted. Once it is trained, this classifier can be used to classify brand new audio instances.

### Noise Filtering

Noise filtering is applied to a new audiofile, which needs denoising. Once the scene classifier identifies the kind of noise evident in the signal, the filter uses the average power spectrum of the identified noise class, and reduces noise accordingly. 

## Installation

Clone this repository. Set the working directory where you clone this repository.

Start a virtual environment:

```
$ python3 -m venv env
$ source env/bin/activate
(env)$
```
Then install necessary installations via pip:
```
(env)$ pip install -r requirements.txt
```
## Before running

Before running, set up the necessary files and variable names.

In the script 'build_my_filter.py', you will see some variable names to define:

```
project_name = ... 
headpath = ...
dataset_directory = ... 
file2filter = ...
```

### 1) Name your project

This program will use the variable 'headpath' to create a directory where new files will be saved. This program creates a subfolder (named with the variable 'project_name') in the headpath, which enables you to create a unique name for your filter project. 

For example, if you set the variable 'headpath' as '/home/my_noise_filters/' and the variable 'project_name' as 'testrun', you will end up with this folder structure:

* home/
    * my_noise_filters/
        * testrun/

It is within the folder 'testrun' that audio feature and model files will be saved.

### 2) Audio Training Data

This program expects you have collected numerous wavfiles to train a deep neural network as a scene classifier. For each class of sounds you want to train it on, create a folder holding corresponding wavfiles. Enter the pathway to these folders into the 'build_my_filter.py' script under the variable name 'audio_classes_dir'.

As an example, let's say you would like to train a network to identify and filter out traffic sounds, air conditioners, and cafe ambient noise. You would collect a  lot of audio wavfiles of each scenario and store them in a folder named as such.

Let's say those folders were stored at the following path: '/home/soundata/dataset/'. The path structure would look something like this:

* home
    * soundata
        * dataset
            * traffic
            * air_conditioner
            * cafe
            

In the 'build_my_filter.py' script, set the variable name 'audio_classes_dir' to be '/home/soundata/dataset/'.

### 3) Wavfile to classify and denoise

As of now, this program expects only wavfiles and does not yet do real-time filtering. Therefore, you need to provide a wavfile that you would like the scene classifier to classify and then the filter to denoise. Enter the pathway of that wavfile under the variable 'soundfile' in the script 'build_my_filter.py'.

### 4) Set features to extract

By default, 'mfcc' features with 40 coefficients/filters are extracted. You can instead extract 'fbank' features if you would like; however for scene classification, 'mfcc' features have proven to work quite well. You can adjust the filters to be higher or lower as you like; keep in mind the higher the number of filters, the higher cost of computation.

## Run 

Once you have set the variables and set up your audio data files in 'build_my_filter.py', run it!

```
(env)$ python3 build_my_filter.py
```

Depending on how many training files you have, duration times vary. I trained this with the DCASE2019 dataset (task A) and it took a good hour to extract the training features, but not long to train the scene classifier.

## File Structure

Once you run this program, it will create the following file structure, depending of course on how you set the variables. 

This is basically the code from 'build_my_filter.py' and below is the folder structure that would result (using default settings):

```
from smartnoisefilter.main import mysmartfilter

project_name = 'testrun' 
headpath = '/home/my_noise_filters/'
dataset_directory = '/home/soundata/dataset/'
file2filter = '/home/example_waves/test.wav'

mysmartfilter(project_name, headpath, dataset_directory,
                sounddata=file2filter)
```

Given the headpath and project_name variables from above, this is how the file 
structure would look:

* home/
    * my_noise_filters/
        * testrun/
            * features/
                * powspec_average/
                    * powspec_settings.csv
                    * powspec_noise_0.npy
                    * powspec_noise_1.npy
                    * powspec_noise_2.npy
                * mfcc_40/
                    * train_data.npy
                    * val_data.npy
                    * test_data.npy
                    * settings_PrepFeatures.csv
                    * output.wav
                    * label_wavefiles.csv
                    * encoded_labels.csv
            * models/
                * mfcc_40/
                    *modelname/
                        * settings_SceneClassifier.csv
                        * modelname.h5
                        * log.csv
                        * bestmodel_modelname.h5

This file structure uses the default settings (i.e. 'mfcc' features, with 40 coefficients). If you extract 'fbank' features, or if you extract a different number of coefficients / filters, you can adjust those in the 'build_my_filter.py' script. New folders with corresponding feature identifiers will be created (e.g. instead of mfcc_40/, it would be fbank_40 or mfcc_13, etc.). This allows you to extract various kinds of features to train your scene classifiers on and see which work best for your data/purpose. 
