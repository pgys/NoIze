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

 
def noizefilter(headpath, filter_project_name, target_wavfile, noise_wavfile=None, 
                scale=1, apply_postfilter=False):
    '''Example code for implementing NoIze as just a noise filter.
    '''
    import noize
    if not noise_wavfile:
        #use background noise to filter out local noise
        output_file = '{}/background_noise.wav'.format(headpath+filter_project_name)
        noize.filtersignal(output_file,target_wavfile,
                           scale=scale,apply_postfilter=apply_postfilter)
        return None
    else:
        #use a separate noise file to reduce noise
        output_file = '{}/separate_noise.wav'.format(headpath+filter_project_name)
        noize.filtersignal(output_file,target_wavfile,noise_file=noise_wavfile,
                        scale=scale,apply_postfilter=apply_postfilter)
        return None

def noizeclassifier(headpath, classifer_project_name, 
                    target_wavfile=None, audiodir=None,
                    feature_type='fbank',audioclass_wavfile_limit=None):
    '''Example code for implementing NoIze as just a sound classifier.
    '''
    import noize
    #extract data
    my_project = noize.PathSetup(classifer_project_name,
                            headpath,
                            audiodir,
                            feature_type=feature_type)
    
    no_conflicts_feats = my_project.cleanup_feats()
    if no_conflicts_feats == False:
        raise FileExistsError('Program cannot run.\
            \nMove the conflicting files or change project name.')
    feat_dir = my_project.feature_dirname
    if my_project.features is True and \
            feat_dir in str(my_project.features_dir) or \
                my_project.features is False:
        print('\nFeatures have been extracted.')
        print('\nLoading corresponding feature settings.')
        feats_class = noize.getfeatsettings(my_project)
    elif audiodir:
        feats_class, my_project = noize.run_featprep(
            my_project,
            feature_type=feature_type,
            limit=audioclass_wavfile_limit)
    import noize.models
    # Has a model been trained and saved?
    if my_project.model is not None \
            and feat_dir in str(my_project.model.parts[-3]):
        print('\nLoading previously trained classifier.')
        classifer_class = noize.models.loadclassifier(my_project)
    else:
        print('\nNow training classifier with train, val, test datasets.')
        # check for file conflicts
        no_conflicts = my_project.cleanup_models()
        if no_conflicts == False:
            raise FileExistsError('Program cannot run.\
                \nMove the conflicting files or change project name.')
        classifer_class = noize.models.buildclassifier(my_project)
    
    if target_wavfile:
        classify = noize.models.ClassifySound(target_wavfile, my_project, feats_class, classifer_class)
        label, label_encoded = classify.get_label()
        print('\nLabel classified: ', label)
