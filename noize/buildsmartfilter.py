#!/bin/bash
# Copyright 2019 Peggy Sylopp und Aislyn Rose GbR
# All rights reserved
# This file is part of the  NoIze-framework
# The NoIze-framework is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by the  
# Free Software Foundation, either version 3 of the License, or (at your option) 
# any later version.
#
#@author Aislyn Rose
#@version 0.1
#@date 31.08.2019
#
# The  NoIze-framework  is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
# details. 
#
# You should have received a copy of the GNU AFFERO General Public License 
# along with the NoIze-framework. If not, see http://www.gnu.org/licenses/.

###############################################################################
import sys
import pathlib

from noize.file_architecture.paths import PathSetup
from noize.acousticfeats_ml.modelfeats import loadfeature_settings, prepfeatures
from noize.models.cnn import ClassifySound, loadclassifier, buildclassifier
from noize.filterfun.filters import calc_audioclass_powerspecs,\
    coll_beg_audioclass_samps
from noize.filterfun.applyfilter import filtersignal


def mysmartfilter(name_dataset, headpath, audio_classes_dir,
                  feature_type='mfcc', num_filters=40,
                  sounddata=None,
                  scale=1, segment_length_ms=1000,
                  apply_postfilter=False,
                  augment_data=False,
                  limit=None,
                  use_rand_noisefile=True,
                  force_label=None,
                  classify_noise=True):
    '''Applies feature prep, model training, and filtering to wavfile.
    '''
    if scale == 0:
        raise ValueError('scale cannot be set to 0')
        sys.exit()
    my_filter = PathSetup(name_dataset,
                          headpath,
                          audio_classes_dir,
                          feature_type=feature_type,
                          num_filters=num_filters,
                          segment_length_ms=segment_length_ms)

    # Have features been extracted?
    # check for file conflicts
    no_conflicts_feats = my_filter.cleanup_feats()
    if no_conflicts_feats == False:
        raise FileExistsError('Program cannot run.\
            \nMove the conflicting files or change project name.')
    feat_dir = my_filter.feature_dirname
    if my_filter.features is True and \
            feat_dir in str(my_filter.features_dir) or \
                my_filter.features is False:
        print('\nFeatures have been extracted.')
        print('\nLoading corresponding feature settings.')
        prep_feats = loadfeature_settings(my_filter)
    elif audio_classes_dir:
        print('\nExtracting dataset features --> train.npy, val.npy, test.npy')
        prep_feats, my_filter = prepfeatures(
            my_filter,
            feature_type=feature_type,
            num_filters=num_filters,
            segment_dur_ms=segment_length_ms,
            limit=limit, augment_data=augment_data)
    # Has the averaged power of the dataset been calculated?
    # check for file conflicts
    no_conflicts_ps = my_filter.cleanup_powspec()
    # conflicts found but were not resolved:
    if no_conflicts_ps is False:
        raise FileExistsError('Program cannot run.\
            \nMove the conflicting files or change project name.')
    # conflicts found but were resolved:
    elif no_conflicts_ps is True:
        my_filter.powspec = None
    if my_filter.powspec is None:
        if not use_rand_noisefile:
            print("\nConducting Welch's method on each class in dataset..")
            calc_audioclass_powerspecs(my_filter, prep_feats, segment_length_ms,
                                    augment_data=augment_data)
        else:
            files_per_audioclass=1
            print("\nSaving beg 120ms from {} files from each audioclass.".format(
                files_per_audioclass))
            coll_beg_audioclass_samps(my_filter, 
                                        prep_feats, 
                                        num_each_audioclass=files_per_audioclass,
                                        dur_ms=1000)
    else:
        print("\nNoise class data extraction already performed on this dataset.")

    # Has a model been trained and saved?
    if my_filter.model is not None \
            and feat_dir in str(my_filter.model.parts[-3]):
        print('\nLoading previously trained scene classifier.')
        scene = loadclassifier(my_filter)
    else:
        print('\nNow training scene classifier with train, val, test datasets.')
        # check for file conflicts
        no_conflicts = my_filter.cleanup_models()
        if no_conflicts == False:
            raise FileExistsError('Program cannot run.\
                \nMove the conflicting files or change project name.')
        scene = buildclassifier(my_filter)

    if classify_noise:
        # Smart filtering begins:
        # Work with new audiofile: classify the background noise and
        # filter it out
        env = ClassifySound(sounddata, my_filter, prep_feats, scene)
        if force_label and isinstance(force_label,str):
            encoded_labels = scene.load_labels()
            for key, value in encoded_labels.items():
                if force_label.lower() == value.lower():
                    label = value
                    label_encoded = key
        else:
            label, label_encoded = env.get_label()
        print('label applied: ', label)
        # load average power spectrum of detected environment
        noise_powspec = env.load_assigned_avepower(label_encoded)
    else:
        noise_powspec = None
    # apply filter:
    if isinstance(sounddata, str):
        sounddata = pathlib.Path(sounddata)
    if isinstance(sounddata, pathlib.PosixPath):
        base_name = sounddata.parts[-1]
    else:
        base_name = 'output.wav'
    
    #adjust filename based on settings
    if apply_postfilter:
        postfilter = 'postfilter_'
    else:
        postfilter = '_'
    if classify_noise:
        label = label
    else:
        label = 'backgroundnoise'
    
    if force_label and classify_noise:
        forced = True
    else:
        forced = False
    
    outputname = 'filtered_scale*{}_{}_{}_forced{}_{}{}'.format(
        scale, name_dataset, label, forced, postfilter, base_name)

    if len(outputname) > 4 and outputname[-4:] == '.wav':
        outputname = outputname
    else:
        outputname = outputname+'.wav'

    filtersignal(
        output_file=my_filter.features_dir.joinpath(outputname),
        wavfile=sounddata,
        noise_file=noise_powspec,
        scale=scale,
        apply_postfilter=apply_postfilter)
    
    return my_filter.features_dir.joinpath(outputname)
