# NoIze

<a href='https://aislynrose.bitbucket.io/'>Documentation</a>

<a href='https://notebooks.ai/a-n-rose/noize-demo-360a3df2'>Demo</a>

This smart noise filter was built under the project \\ \\NoIze/ /, managed by Peggy Sylopp (contact@peggy-sylopp.net).

Author of the code/software in this repository: Aislyn Rose (rose.aislyn.noelle@gmail.com, <a href='https://a-n-rose.github.io/'>a-n-rose.github.io</a>)

## About

This software has functionality for noise filtering, sound classification, and smart noise filtering. <a href='https://aislynrose.bitbucket.io/readme.html#a-walk-through-the-modules'>Here</a> you can access a more detailed walkthrough of the smart noise filter functionality.

# NoIze as a smart noise filter

This requires a bit of effort because data collection is necessary for this to work. 

### Collect some data

The smart filter needs data for training. 

Collect wavfiles of different noise classes you would like to filter out. 

For this example, let's say you've collected wavfiles belonging to the audio classes 'traffic', 'cafe', and 'train'. Save them in a directory that has such a structure:

##### Figure 1:
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

##### Figure 2:
![Imgur](https://i.imgur.com/WBoZFkk.png)

For more on what these files do, <a href='https://aislynrose.bitbucket.io/readme.html#file-setup'>here</a> you can find a more detailed description.

### Making smart filter adjustments

One can increase/decrease the `scale` of the filter and apply a postfilter (`apply_postfilter`). Additionally, one can force the smart filter to use a different label than the one automatically classified by the smart filter (`force_label`). One can also decide to not implement the 'smart' part of the filter and just use the background noise from the file to reduce the noise (`classify_noise`). 
```
(env)..$ from noize.buildsmartfilter import mysmartfilter

(env)..$ project_name = 'test_smartfilter'
(env)..$ headpath = 'directory_where_createdfiles_should_be_saved'
(env)..$ audio_classes_dir = 'directory_where_training_data_is_located'

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
This will generate a similar structure to the file architecture in the last figure; however, instead of folders with 'mfcc', you would see folders with 'fbank' instead. MFCC features are set as default because they have proven quite successful in acoustic scene / noise classification.

## Structure of the smart filter 

Once the sound classifier has been trained for the smart filter, the smart filter runs as follows:

The upper case letters refer to functionality.
The lower case letters refer to the major Python libraries used.
![Imgur](https://i.imgur.com/gsSfAtD.png)


# NoIze as a simple filter

```
(env)..$ import noize
```

## Use the noisy signal's background noise for filtering:

The filtered signal will be saved under the `output_file` path.

```
(env)..$ noize.filtersignal(output_file = 'name_filteredsignal.wav, 
                            target_file = 'noisysignal.wav')
```
## Use a separate noise file for filtering:

```
(env)..$ noize.filtersignal(output_file = 'name_filteredsignal.wav, 
                            target_file = 'noisysignal.wav', 
                            noise_file = 'backgroundnoise.wav')
```
## Increase or decrease the scale of the filter:

Default is 1 and can be set to just about any number except 0. 

### Decrease:
```
(env)..$ noize.filtersignal(output_file = 'name_filteredsignal.wav, 
                            target_file = 'noisysignal.wav', 
                            scale = 0.5)
```
### Increase:
```
(env)..$ noize.filtersignal(output_file = 'name_filteredsignal.wav, 
                            target_file = 'noisysignal.wav', 
                            scale = 1.5)
```
## Apply post filter to decrease 'musical noise' / distortion:

Default is False
```
(env)..$ noize.filtersignal(output_file = 'name_filteredsignal.wav, 
                            target_file = 'noisysignal.wav', 
                            apply_postfilter = True)
```

# NoIze as a sound classifier

Data collection is necessary for this to work. You can train this classifier on the <a href='https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html'>speech commands dataset</a>, acoustic scenes, healthy vs clinical speech, speech vs non-speech (i.e. silence or noise). It is not specific to noise classification. 

The structure of the training data needs to be similar to Figure 1.

## Just building a classifier with defaults:
```
(env)..$ from noize.templates import noizeclassifier

(env)..$ project_name = 'test_soundclassifier'
(env)..$ headpath = 'directory_where_createdfiles_should_be_saved'
(env)..$ audio_classes_dir = 'directory_where_training_data_is_located'

(env)..$ noizeclassifier(project_name,
                        headpath,
                        audio_classes_dir)
```
This will train and save a classifier in the created models directory (see Figure 2).

## Adjusting the settings:

```
(env)..$ from noize.templates import noizeclassifier

(env)..$ project_name = 'test_soundclassifier'
(env)..$ headpath = 'directory_where_createdfiles_should_be_saved'
(env)..$ audio_classes_dir = 'directory_where_training_data_is_located'

(env)..$ noizeclassifier(project_name,
                        headpath,
                        audio_classes_dir,
                        feature_type = 'mfcc', # default = 'fbank'
                        target_wavfile = 'file2classify.wav', # default = None
                        audioclass_wavfile_limit = 120, # useful for balancing classes
                        )
```
This will not only train and save a classifier (if one doesn't already exist), but will also classify the `target_wavfile`. The `feature_type` concerns which features are extracted from the training data. Options: 'mfcc' or 'fbank'. The default is set to FBANK, as the architecture of the classifier is based on that used in the paper by <a href='https://ieeexplore.ieee.org/abstract/document/8278160'>Sehgal and Kehtarnavaz (2017)</a>. In general, FBANK features tend to work better in speech/ voice related tasks than MFCCs. However, it is useful to be able to see which is better, which one can explore here. The `audioclass_wavfile_limit` is to allow for a bit more control if you have many more wavfiles in one audio class than another.

# Installation

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

# References

A. Sehgal and N. Kehtarnavaz, "A Convolutional Neural Network Smartphone App for Real-Time Voice Activity Detection," in IEEE Access, vol. 6, pp. 9017-9026, 2018.
doi: 10.1109/ACCESS.2018.2800728
 
