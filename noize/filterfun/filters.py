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

'''
The filters module covers functions related to the filtering out of noise of
a target signal, whether that be the collection of power spectrum values to
calculate the average power spectrum of each audio class or to measure 
signal-to-noise ratio of a signal and ultimately the the actual filtering 
process.
'''
###############################################################################
import numpy as np

import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
noizedir = os.path.dirname(parentdir)
sys.path.insert(0, noizedir)

import noize

# what Wiener Filter and Average pow spec can inherit
class FilterSettings:
    def __init__(self,
                 frame_duration_ms=20,
                 percent_overlap=0.5,
                 sampling_rate=48000,
                 window_type='hamming'):
        self.frame_dur = frame_duration_ms
        self.samplerate = sampling_rate
        self.frame_length = noize.dsp.calc_frame_length(
            frame_duration_ms,
            sampling_rate)
        self.percent_overlap = percent_overlap
        self.overlap_length = noize.dsp.calc_num_overlap_samples(
            self.frame_length,
            percent_overlap)
        self.window_type = window_type
        self.window = noize.dsp.create_window(window_type, self.frame_length)
        self.num_fft_bins = self.frame_length


class WienerFilter(FilterSettings):
    """Interactive class to explore Wiener filter settings on audio signals.

    These class methods implement research based algorithms with low 
    computational cost, aimed for noise reduction via mobile phone

    Attributes
    ----------
    frame_dur : int, float
        Time in milliseconds of each audio frame window (default 20)
    sampling_rate : int 
        Desired sampling rate of audio; audio will be resampled to match if
        audio has other sampling rate (default 48000)
    frame_length : int 
        Number of audio samples in each frame: frame_dur multiplied with
        sampling_rate, divided by 1000 (default 960)
    overlap_length : int 
        Number of overlapping audio samples between subsequent frames: 
        frame_length multiplied by percent_overlap, floored (default 480)
    beta : float
        Value applied in Wiener filter that smooths the application of gain;
        default set according to previous research (default 0.98).
    window_type : str
        Type of window applied to audio frames: hann vs hamming (default
        'hamming')
    window : ndarray
        The window according to indicated window_type and frame_length; this 
        value can be applied directly to a frame of audio samples
    """

    def __init__(self,
                 smooth_factor=0.98,
                 first_iter=None,
                 max_vol = 0.4):
        FilterSettings.__init__(self)
        self.beta = smooth_factor
        self.first_iter = first_iter
        self.noise_subframes = None
        self.gain = None
        self.max_vol = max_vol

    def get_samples(self, wavfile, dur_sec=None):
        """Load signal and save original volume

        Parameters
        ----------
        wavfile : str
            Path and name of wavfile to be loaded
        dur_sec : int, float optional
            Max length of time in seconds (default None)

        Returns 
        ----------
        samples : ndarray
            Array containing signal amplitude values in time domain
        """
        samples, sr = noize.dsp.load_signal(
            wavfile, self.samplerate, dur_sec=dur_sec)
        self.set_volume(samples, max_vol = self.max_vol)
        return samples

    def set_volume(self, samples, max_vol = 0.4, min_vol = 0.15):
        """Records and limits the maximum amplitude of original samples 

        This enables the output wave to be within a range of
        volume that does not go below or too far above the 
        orignal maximum amplitude of the signal. 

        Parameters
        ----------
        samples : ndarray
            The original samples of a signal (1 dimensional), of any length
        max_vol : float
            The maximum volume level. If a signal has values higher than this 
            number, the signal is curtailed to remain at and below this number.
        min_vol : float
            The minimum volume level. If a signal has only values lower than
            this number, the signal is amplified to be at this number and below.
        
        Returns
        -------
        None
        """
        if isinstance(samples, np.ndarray):
            max_amplitude = samples.max()
        else:
            max_amplitude = max(samples)
        self.vol_orig = max_amplitude
        if max_amplitude > max_vol:
            self.max_vol = max_vol
        elif max_amplitude < min_vol:
            self.max_vol = min_vol
        else:
            self.max_vol = max_amplitude
        return None

    def set_num_subframes(self, len_samples, is_noise=False):
        """Sets the number of target or noise subframes available for processing

        Parameters
        ----------
        len_samples : int 
            The total number of samples in a given signal
        is_noise : bool
            If False, subframe number saved under self.target_subframes, otherwise 
            self.noise_subframes (default False)

        Returns
        -------
        None
        """
        if is_noise:
            self.noise_subframes = noize.dsp.calc_num_subframes(
                tot_samples=len_samples,
                frame_length=self.frame_length,
                overlap_samples=self.overlap_length
            )
        else:
            self.target_subframes = noize.dsp.calc_num_subframes(
                tot_samples=len_samples,
                frame_length=self.frame_length,
                overlap_samples=self.overlap_length
            )
        return None

    def load_power_vals(self, path_npy):
        """Loads and checks shape compatibility of averaged power values

        Parameters
        ----------
        path_npy : str, pathlib.PosixPath
            Path to .npy file containing power information. 

        Returns
        -------
        power_values : ndarray
            The power values as long as they have the shape (self.num_fft_bins, 1)
        """
        power_values = noize.paths.load_feature_data(path_npy)
        if power_values.shape[0] != self.num_fft_bins:
            raise ValueError("Power value shape does not match settings.\
                \nProvided power value shape: {}\
                \nExpected shape: ({},)".format(
                power_values.shape, self.num_fft_bins))
        # get rid of extra, unnecessary dimension
        if power_values.shape == (self.num_fft_bins, 1):
            power_values = power_values.reshape(self.num_fft_bins,)
        return power_values

    def check_volume(self, samples):
        """ensures volume of filtered signal is within the bounds of the original
        """
        max_orig = round(max(samples), 2)
        samples = noize.dsp.control_volume(samples, self.max_vol)
        max_adjusted = round(max(samples), 2)
        if max_orig != max_adjusted:
            print("volume adjusted from {} to {}".format(max_orig, max_adjusted))
        return samples

    def save_filtered_signal(self, output_file, samples, overwrite=False):
        saved, filename = noize.paths.save_wave(
            output_file, samples, self.samplerate, overwrite=overwrite)
        if saved:
            print('Wavfile saved under: {}'.format(filename))
            return True
        else:
            print('Error occurred. {} not saved.'.format(filename))
            return False


class WelchMethod(FilterSettings):
    def __init__(self,
                 len_noise_sec=1):
        FilterSettings.__init__(self)
        self.len_noise_sec = len_noise_sec

    def set_num_subframes(self, len_samples, noise=True):
        '''calculate and set number of subframes required to process total samples
        '''
        if noise:
            self.noise_subframes = noize.dsp.calc_num_subframes(tot_samples=len_samples,
                                                          frame_length=self.frame_length,
                                                          overlap_samples=self.overlap_length)
        else:
            self.target_subframes = noize.dsp.calc_num_subframes(tot_samples=len_samples,
                                                           frame_length=self.frame_length,
                                                           overlap_samples=self.overlap_length)
        return None

    def get_power(self, samples, matrix2store_power):
        section_start = 0
        for frame in range(self.noise_subframes):
            noise_sect = samples[section_start:section_start +
                                 self.frame_length]
            noise_w_win = noize.dsp.apply_window(noise_sect, self.window)
            noise_fft = noize.dsp.calc_fft(noise_w_win)
            noise_power = noize.dsp.calc_power(noise_fft)
            for i, row in enumerate(noise_power):
                matrix2store_power[i] += row
            section_start += self.overlap_length
        return matrix2store_power

    def coll_pow_average(self, wave_list, scale=None, augment_data=False):
        pwspec_shape = self.window.shape+(1,)
        noise_powspec = noize.matrixfun.create_empty_matrix(
            pwspec_shape, complex_vals=False)
        for j, wav in enumerate(wave_list):
            n, sr = noize.dsp.load_signal(wav, dur_sec=self.len_noise_sec)
            if augment_data:
                samples = noize.augmentdata.spread_volumes(n)
            else:
                samples = (n,)
            for sampledata in samples:
                if scale:
                    sampledata *= scale
                noise_powspec = self.get_power(sampledata, noise_powspec)

            progress = (j+1) / len(wave_list) * 100
            sys.stdout.write(
                "\r%d%% through sound class wavfile list" % progress)
            sys.stdout.flush()
        print('\nFinished\n')
        tot_powspec_collected = self.noise_subframes * \
            len(wave_list) * len(samples)
        noise_powspec = noize.dsp.calc_average_power(
            noise_powspec, tot_powspec_collected)
        return noise_powspec

def get_average_power(class_waves_dict, encodelabel_dict,
                      powspec_dir, duration_sec=1, augment_data=False):
    avspec = WelchMethod(len_noise_sec=duration_sec)
    avspec.set_num_subframes(int(avspec.len_noise_sec * avspec.samplerate),
                             noise=True)
    total_classes = len(class_waves_dict)
    count = 0
    for key, value in class_waves_dict.items():
        print('\nProcessing class {} out of {}'.format(count+1, total_classes))
        # value = str(waves list)
        wave_list = noize.paths.string2list(value)
        noise_powspec = avspec.coll_pow_average(wave_list,
                                                augment_data=augment_data)
        path2save = powspec_dir.joinpath(
            # key is str label of class--> encoded integer
            'powspec_noise_{}.npy'.format(encodelabel_dict[key]))
        noize.paths.save_feature_data(path2save, noise_powspec)
        count += 1
    return None

def calc_audioclass_powerspecs(filter_class, feature_class, dur_ms,
                               augment_data=False):
    class_waves_dict = noize.paths.load_dict(filter_class.labels_waves_path)
    labels_encoded_dict = noize.paths.load_dict(filter_class.labels_encoded_path)
    encodelabel_dict = {}
    for key, value in labels_encoded_dict.items():
        encodelabel_dict[value] = key
    total_classes = len(class_waves_dict)
    fs = FilterSettings()
    powspec_settings = fs.__dict__
    # add number of classes to settings dictionary to check all classes get processed
    powspec_settings['num_audio_classes'] = total_classes
    powspec_settings['processing_window_sec'] = dur_ms/1000.
    powspec_settings_filename = filter_class.powspec_path.joinpath(
        filter_class._powspec_settings)
    noize.paths.save_dict(powspec_settings, powspec_settings_filename)

    get_average_power(class_waves_dict, encodelabel_dict,
                      filter_class.powspec_path,
                      duration_sec=dur_ms/1000.0,
                      augment_data=augment_data)
    return None

def coll_beg_audioclass_samps(
    filter_class, feature_class, num_each_audioclass=1, dur_ms=500):
    class_waves_dict = noize.paths.load_dict(filter_class.labels_waves_path)
    labels_encoded_dict = noize.paths.load_dict(filter_class.labels_encoded_path)
    encodelabel_dict = {}
    for key, value in labels_encoded_dict.items():
        encodelabel_dict[value] = key
    for key, value in class_waves_dict.items():
        wavlist = noize.paths.string2list(value)
        label_int = encodelabel_dict[key]
        rand_indices = np.random.randint(
            0,len(class_waves_dict),num_each_audioclass)
        noisefiles = []
        for index in rand_indices:
            noisefiles.append(wavlist[index])
        get_save_begsamps(noisefiles, label_int,
                          filter_class.powspec_path,
                          samplerate=feature_class.sr,
                          dur_ms=dur_ms)
    return None

def get_save_begsamps(wavlist,audioclass_int,
                      powspec_dir,samplerate=48000,dur_ms=1000):
    numsamps = noize.dsp.calc_frame_length(dur_ms,samplerate)
    for i, wav in enumerate(wavlist):
        y, sr = noize.dsp.load_signal(wav,sampling_rate=samplerate)
        y120 = y[:numsamps]
        filename = powspec_dir.joinpath(
            'beg120ms{}sr_{}_audioclass{}.npy'.format(samplerate, i,audioclass_int))
        noize.paths.save_feature_data(filename, y120)
    return None
