#!/bin/bash
# Copyright 2019 Peggy Sylopp und Aislyn Rose GbR
# All rights reserved
# This file is part of the  NoIze-framework
# The NoIze-framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
#@author Aislyn Rose
#@version 0.1
#@date 31.08.2019
#
# The  NoIze-framework  is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. 
#
# You should have received a copy of the GNU AFFERO General Public License along with the NoIze-framework. If not, see http://www.gnu.org/licenses/.

import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from noize.acousticfeats_ml import featorg
import pathlib
import pytest



def test_make_number_str2int():
    str2int = 9
    value = featorg.make_number('9')
    assert str2int == value


def test_make_number_str2float():
    str2float = 9.0
    value = featorg.make_number('9.0')
    assert str2float == value


def test_make_number_str2none():
    str2none = None
    value = featorg.make_number('')
    assert str2none == value


def test_make_number_str2str():
    str2str = '9e'
    value = featorg.make_number('9e')
    assert str2str == value


def test_make_number_int2int():
    int2int = 9
    value = featorg.make_number(9)
    assert int2int == value


def test_make_number_float2float():
    float2float = 9.0
    value = featorg.make_number(9.0)
    assert float2float == value


def test_create_label2audio_dict_lists():
    PosixPath = pathlib.PosixPath
    label2audio_dict = {'fridge': [PosixPath('data/audio/fridge/fridge1.wav')],
                        'wind': [PosixPath('data/audio/wind/wind1.wav')],
                        'vacuum': [PosixPath('data/audio/vacuum/vacuum1.wav'),
                                   PosixPath('data/audio/vacuum/vacuum2.wav')]}
    labels = ['vacuum', 'fridge', 'wind']
    paths = [pathlib.Path('data/audio/vacuum/vacuum1.wav'), pathlib.Path('data/audio/fridge/fridge1.wav'),
             pathlib.Path('data/audio/vacuum/vacuum2.wav'), pathlib.Path('data/audio/wind/wind1.wav')]
    value_dict = featorg.create_label2audio_dict(labels, paths)
    assert label2audio_dict == value_dict


def test_create_label2audio_dict_sets():
    PosixPath = pathlib.PosixPath
    label2audio_dict = {'fridge': [PosixPath('data/audio/fridge/fridge1.wav')],
                        'wind': [PosixPath('data/audio/wind/wind1.wav')],
                        'vacuum': [PosixPath('data/audio/vacuum/vacuum1.wav'),
                                   PosixPath('data/audio/vacuum/vacuum2.wav')]}
    labels = set(['vacuum', 'fridge', 'wind'])
    paths = set([pathlib.Path('data/audio/vacuum/vacuum1.wav'), pathlib.Path('data/audio/fridge/fridge1.wav'),
                 pathlib.Path('data/audio/vacuum/vacuum2.wav'), pathlib.Path('data/audio/wind/wind1.wav')])
    value_dict = featorg.create_label2audio_dict(labels, paths)
    assert label2audio_dict == value_dict


def test_create_label2audio_dict_set_and_list():
    PosixPath = pathlib.PosixPath
    label2audio_dict = {'fridge': [PosixPath('data/audio/fridge/fridge1.wav')],
                        'wind': [PosixPath('data/audio/wind/wind1.wav')],
                        'vacuum': [PosixPath('data/audio/vacuum/vacuum1.wav'),
                                   PosixPath('data/audio/vacuum/vacuum2.wav')]}
    labels = set(['vacuum', 'fridge', 'wind'])
    paths = [pathlib.Path('data/audio/vacuum/vacuum1.wav'), pathlib.Path('data/audio/fridge/fridge1.wav'),
             pathlib.Path('data/audio/vacuum/vacuum2.wav'), pathlib.Path('data/audio/wind/wind1.wav')]
    value_dict = featorg.create_label2audio_dict(labels, paths)
    assert label2audio_dict == value_dict


def test_create_label2audio_dict_list_and_set():
    PosixPath = pathlib.PosixPath
    label2audio_dict = {'fridge': [PosixPath('data/audio/fridge/fridge1.wav')],
                        'wind': [PosixPath('data/audio/wind/wind1.wav')],
                        'vacuum': [PosixPath('data/audio/vacuum/vacuum1.wav'),
                                   PosixPath('data/audio/vacuum/vacuum2.wav')]}
    labels = list(['vacuum', 'fridge', 'wind'])
    paths = set([pathlib.Path('data/audio/vacuum/vacuum1.wav'), pathlib.Path('data/audio/fridge/fridge1.wav'),
                 pathlib.Path('data/audio/vacuum/vacuum2.wav'), pathlib.Path('data/audio/wind/wind1.wav')])
    value_dict = featorg.create_label2audio_dict(labels, paths)
    assert label2audio_dict == value_dict


def test_create_label2audio_dict_typeerror_labels():
    labels = str(set(['vacuum', 'fridge', 'wind']))
    paths = [pathlib.Path('data/audio/vacuum/vacuum1.wav'), pathlib.Path('data/audio/fridge/fridge1.wav'),
             pathlib.Path('data/audio/vacuum/vacuum2.wav'), pathlib.Path('data/audio/wind/wind1.wav')]
    with pytest.raises(TypeError):
        featorg.create_label2audio_dict(labels, paths)


def test_create_label2audio_dict_typeerror_paths():
    labels = set(['vacuum', 'fridge', 'wind'])
    paths = str([pathlib.Path('data/audio/vacuum/vacuum1.wav'), pathlib.Path('data/audio/fridge/fridge1.wav'),
                 pathlib.Path('data/audio/vacuum/vacuum2.wav'), pathlib.Path('data/audio/wind/wind1.wav')])
    with pytest.raises(TypeError):
        featorg.create_label2audio_dict(labels, paths)


def test_create_label2audio_dict_valueerror_no_matching_label():
    labels = set(['vacuum', 'fridge', 'wind'])
    paths = [pathlib.Path('data/audio/vac/vac1.wav'), pathlib.Path('data/audio/fridg/fridg1.wav'),
             pathlib.Path('data/audio/vac/vac2.wav'), pathlib.Path('data/audio/win/win1.wav')]
    with pytest.raises(ValueError):
        featorg.create_label2audio_dict(labels, paths)


def test_create_label2audio_dict_limit():
    PosixPath = pathlib.PosixPath
    label2audio_dict = {
        'vacuum': [PosixPath('data/audio/vacuum/vacuum1.wav'),
                   PosixPath('data/audio/vacuum/vacuum2.wav'),
                   PosixPath('data/audio/vacuum/vacuum4.wav')],
        'fridge': [PosixPath('data/audio/fridge/fridge1.wav'),
                   PosixPath('data/audio/fridge/fridge2.wav')],
        'wind': [PosixPath('data/audio/wind/wind1.wav'),
                 PosixPath('data/audio/wind/wind2.wav'),
                 PosixPath('data/audio/wind/wind3.wav')]}
    labels = ['vacuum', 'fridge', 'wind']
    paths = [pathlib.Path('data/audio/vacuum/vacuum1.wav'),
             pathlib.Path('data/audio/fridge/fridge1.wav'),
             pathlib.Path('data/audio/vacuum/vacuum2.wav'),
             pathlib.Path('data/audio/wind/wind1.wav'),
             pathlib.Path('data/audio/vacuum/vacuum3.wav'),
             pathlib.Path('data/audio/fridge/fridge2.wav'),
             pathlib.Path('data/audio/vacuum/vacuum4.wav'),
             pathlib.Path('data/audio/wind/wind2.wav'),
             pathlib.Path('data/audio/wind/wind3.wav')]
    value_dict = featorg.create_label2audio_dict(labels, paths, limit=3)
    assert label2audio_dict == value_dict


def test_create_label2audio_dict_seed():
    PosixPath = pathlib.PosixPath
    label2audio_dict = {
        'vacuum': [PosixPath('data/audio/vacuum/vacuum2.wav'),
                   PosixPath('data/audio/vacuum/vacuum3.wav'),
                   PosixPath('data/audio/vacuum/vacuum4.wav')],
        'fridge': [PosixPath('data/audio/fridge/fridge1.wav'),
                   PosixPath('data/audio/fridge/fridge2.wav')],
        'wind': [PosixPath('data/audio/wind/wind1.wav'),
                 PosixPath('data/audio/wind/wind2.wav'),
                 PosixPath('data/audio/wind/wind3.wav')]}
    labels = ['vacuum', 'fridge', 'wind']
    paths = [pathlib.Path('data/audio/vacuum/vacuum1.wav'),
             pathlib.Path('data/audio/fridge/fridge1.wav'),
             pathlib.Path('data/audio/vacuum/vacuum2.wav'),
             pathlib.Path('data/audio/wind/wind1.wav'),
             pathlib.Path('data/audio/vacuum/vacuum3.wav'),
             pathlib.Path('data/audio/fridge/fridge2.wav'),
             pathlib.Path('data/audio/vacuum/vacuum4.wav'),
             pathlib.Path('data/audio/wind/wind2.wav'),
             pathlib.Path('data/audio/wind/wind3.wav')]
    value_dict = featorg.create_label2audio_dict(
        labels, paths, limit=3, seed=15)
    assert label2audio_dict == value_dict


def test_create_dicts_labelsencoded_set():
    label2int = {'air_conditioner': 0, 'fridge': 1, 'wind': 2}
    int2label = {0: 'air_conditioner', 1: 'fridge', 2: 'wind'}
    input_labels = {'wind', 'air_conditioner', 'fridge'}
    value_dict1, value_dict2 = featorg.create_dicts_labelsencoded(input_labels)
    assert label2int == value_dict1
    assert int2label == value_dict2


def test_create_dicts_labelsencoded_list():
    label2int = {'air_conditioner': 0, 'fridge': 1, 'wind': 2}
    int2label = {0: 'air_conditioner', 1: 'fridge', 2: 'wind'}
    input_labels = list({'wind', 'air_conditioner', 'fridge'})
    value_dict1, value_dict2 = featorg.create_dicts_labelsencoded(input_labels)
    assert label2int == value_dict1
    assert int2label == value_dict2


def test_create_dicts_labelsencoded_typeerror():
    input_labels = str({'wind', 'air_conditioner', 'fridge'})
    with pytest.raises(TypeError):
        featorg.create_dicts_labelsencoded(input_labels)


def test_waves2dataset_defaults():
    audiolist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    train, val, test = ([5, 4, 9, 2, 3, 10, 1, 6], [8], [7])
    value_list1, value_list2, value_list3 = featorg.waves2dataset(audiolist)
    assert train == value_list1
    assert val == value_list2
    assert test == value_list3


def test_waves2dataset_train_perc_50():
    audiolist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    train, val, test = ([5, 4, 9, 2, 3, 10], [1, 6], [8, 7])
    value_list1, value_list2, value_list3 = featorg.waves2dataset(
        audiolist,
        train_perc=50)
    assert train == value_list1
    assert val == value_list2
    assert test == value_list3


def test_waves2dataset_seed_0():
    audiolist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    train, val, test = ([7, 1, 2, 5, 6, 9, 10, 8], [4], [3])
    value_list1, value_list2, value_list3 = featorg.waves2dataset(
        audiolist,
        seed=0)
    assert train == value_list1
    assert val == value_list2
    assert test == value_list3


def test_waves2dataset_seed_none():
    audiolist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    train, val, test = ([7, 1, 2, 5, 6, 9, 10, 8], [4], [3])
    value_list1, value_list2, value_list3 = featorg.waves2dataset(
        audiolist,
        seed=None)
    assert train != value_list1
    assert val != value_list2
    assert test != value_list3
    assert len(train) == len(value_list1)
    assert len(val) == len(value_list2)
    assert len(test) == len(value_list3)


def test_waves2dataset_indexerror_emptylist_as_input():
    audiolist = []
    with pytest.raises(IndexError):
        featorg.waves2dataset(audiolist)


def test_waves2dataset_indexerror_str_as_input():
    audiolist = 'can this string be separated into datasets?'
    with pytest.raises(IndexError):
        featorg.waves2dataset(audiolist)


def test_audio2datasets_valueerror_over100percent():
    audio_classes_dir = 'home/audiodataset/'
    encoded_labels_path = 'home/project/labelsencoded.csv'
    label_wavefiles_path = 'home/project/labelwaves.csv'
    perc_train = 200
    with pytest.raises(ValueError):
        featorg.audio2datasets(audio_classes_dir,
                               encoded_labels_path,
                               label_wavefiles_path,
                               perc_train)


def test_audio2datasets_valueerror_train_too_small():
    audio_classes_dir = 'home/audiodataset/'
    encoded_labels_path = 'home/project/labelsencoded.csv'
    label_wavefiles_path = 'home/project/labelwaves.csv'
    perc_train = 1.5
    with pytest.raises(ValueError):
        featorg.audio2datasets(audio_classes_dir,
                               encoded_labels_path,
                               label_wavefiles_path,
                               perc_train)
