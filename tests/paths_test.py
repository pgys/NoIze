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
from noize.file_architecture import paths as pathorg
import pathlib
import pytest


def test_check_extension_add_ext_no_period():
    filename = 'data.npy'
    value = pathorg.check_extension('data', 'npy')
    assert filename == value


def test_check_extension_add_ext_with_period():
    filename = 'data.npy'
    value = pathorg.check_extension('data', '.npy')
    assert filename == value


def test_check_extension_no_change():
    filename = 'data.npy'
    value = pathorg.check_extension('data.npy', 'npy')
    assert filename == value


def test_check_extension_replace_extension():
    filename = 'data.npy'
    value = pathorg.check_extension('data.txt', 'npy', replace=True)
    assert filename == value


def test_check_extension_no_replace_extension():
    filename = 'data.txt.npy'
    value = pathorg.check_extension('data.txt', 'npy', replace=False)
    assert filename == value


def test_check_extension_typeerror_filename():
    with pytest.raises(TypeError):
        pathorg.check_extension(9, 'npy')


def test_check_extension_typeerror_extension():
    with pytest.raises(TypeError):
        pathorg.check_extension('data.txt', ['.npy'])


def test_is_audio_ext_allowed_extension_no_period():
    assert pathorg.is_audio_ext_allowed('wav')


def test_is_audio_ext_allowed_extension_with_period():
    assert pathorg.is_audio_ext_allowed('.wav')


def test_is_audio_ext_allowed_filename_wav():
    assert pathorg.is_audio_ext_allowed('happy.wav')


def test_is_audio_ext_allowed_filename_mp3():
    assert not pathorg.is_audio_ext_allowed('happy.mp3')


def test_is_audio_ext_allowed_filename_ogg():
    assert not pathorg.is_audio_ext_allowed('happy.ogg')


def test_is_audio_ext_allowed_filename_mf4():
    assert not pathorg.is_audio_ext_allowed('happy.m4a')


def test_is_audio_ext_allowed_not_allowed_filename():
    assert not pathorg.is_audio_ext_allowed('happy.aiff')


def test_string2list_posixpath():
    PosixPath = pathlib.PosixPath
    input_str = "[PosixPath('data/audio/vacuum/vacuum1.wav')]"
    output_list = [PosixPath('data/audio/vacuum/vacuum1.wav')]
    value_list = pathorg.string2list(input_str)
    assert output_list == value_list


def test_string2list_pureposixpath():
    PosixPath = pathlib.PosixPath
    input_str = "[PurePosixPath('data/audio/vacuum/vacuum1.wav')]"
    output_list = [PosixPath('data/audio/vacuum/vacuum1.wav')]
    value_list = pathorg.string2list(input_str)
    assert output_list == value_list


def test_string2list_string():
    PosixPath = pathlib.PosixPath
    input_str = "[('data/audio/vacuum/vacuum1.wav')]"
    output_list = [PosixPath('data/audio/vacuum/vacuum1.wav')]
    value_list = pathorg.string2list(input_str)
    assert output_list == value_list
