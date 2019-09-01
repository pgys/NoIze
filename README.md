# smartnoisefilter: NoIze

<a href='https://aislynrose.bitbucket.io/'>Documentation</a>

![Imgur](https://i.imgur.com/YIYRHxK.png)

This smart noise filter was built under the project \\NoIze//, managed by Peggy Sylopp (contact@peggy-sylopp.net).

Author of the code/software in this repository: Aislyn Rose (rose.aislyn.noelle@gmail.com, a-n-rose.github.io)

## About

This software has four main parts: 1) pathway formation, 2) feature extraction, 3) scene classifier training, and 4) noise filtering. 

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
(env)..$
```
Then install necessary installations via pip:
```
(env)..$ pip install -r requirements.txt
```
## NoIze as a simple filter

```
(env)..$ import noize
```

### Use the noisy signal's background noise for filtering:

The filtered signal will be saved under the `output_file` path.

```
(env)..$ noize.filtersignal(output_file = 'name_filteredsignal.wav, 
                            target_file = 'noisysignal.wav')
```
### Use a separate noise file for filtering:

```
(env)..$ noize.filtersignal(output_file = 'name_filteredsignal.wav, 
                            target_file = 'noisysignal.wav', 
                            noise_file = 'backgroundnoise.wav')
```
### Increase or decrease the scale of the filter:

Default is 1 and can be set to just about any number except 0. 

#### Decrease:
```
(env)..$ noize.filtersignal(output_file = 'name_filteredsignal.wav, 
                            target_file = 'noisysignal.wav', 
                            scale = 0.5)
```
#### Increase:
```
(env)..$ noize.filtersignal(output_file = 'name_filteredsignal.wav, 
                            target_file = 'noisysignal.wav', 
                            scale = 1.5)
```
### Apply post filter to decrease 'musical noise' / distortion:

Default is False
```
(env)..$ noize.filtersignal(output_file = 'name_filteredsignal.wav, 
                            target_file = 'noisysignal.wav', 
                            apply_postfilter = True)
```

## NoIze as a smart noise filter

For this, a bit more effort is necessary. 

### Collect some data

The smart filter needs data for training. 

Collect wavfiles of different noise classes you would like to filter out. For this example, let's say you've collected wavfiles belonging to the audio classes 'traffic', 'cafe', and 'train'. Save them in a directory that has such a structure:

![Imgur](https://i.imgur.com/ycCLuUN.png)

### Run the smart filter

Once you've collected data, you can run the program. Running code like the following example generates a file structure, similar to that showed in the figure below.

### Smart filter with defaults
```
(env)..$ from noize.buildsmartfilter import mysmartfilter

(env)..$ project_name = 'test_smartfilter'
(env)..$ headpath = 'directory_where_createdfiles_should_be_saved'
(env)..$ audio_classes_dir = 'directory_where_training_data_is_located'

(env)..$ filteredwavfile = mysmartfilter(project_name,
                                headpath,
                                audio_classes_dir,
                                sounddata = 'noisysignal.wav')
```
The `filteredwavfile` is the filename where the filtered signal is stored.

![Imgur](https://i.imgur.com/WBoZFkk.png)

For more on what these files do, <a href='https://aislynrose.bitbucket.io/readme.html#a-walk-through-the-modules'>here</a> you can find a more detailed description.

### Making smart filter adjustments

Similar to the simple filter application presented above, one can increase/decrease the scale of the filter and apply a postfilter. Additionally, one can force the smart filter to use a different label than the one automatically classified by the smart filter. One can also decide to not implement the 'smart' part of the filter and just use the background noise from the file to reduce the noise. 
```
(env)..$ from noize.buildsmartfilter import mysmartfilter

(env)..$ project_name = 'test_smartfilter'
(env)..$ headpath = 'directory_where_createdfiles_should_be_saved'

(env)..$ filteredwavfile = mysmartfilter(project_name,
                                headpath,
                                audio_classes_dir,
                                sounddata = 'noisysignal.wav',
                                feature_type = 'fbank', # default = 'mfcc'
                                scale = 0.5, # defaul = 1
                                apply_postfilter = True, # default False
                                force_label = 'traffic', # enter the label to be applied
                                classify_noise = False, # default True
                                )
```
This will generate a similar structure to the file architecture in the last figure; however, instead of folders with 'mfcc', you would see folders with 'fbank' instead.
