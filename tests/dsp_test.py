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
from noize.mathfun import matrixfun
from noize.mathfun import dsp
import noize
import numpy as np
import pytest



def test_calc_frame_length():
    frame_length = 20
    value = dsp.calc_frame_length(dur_frame_millisec=20, sampling_rate=1000)
    assert value == frame_length


def test_calc_num_overlap_samples():
    overlap_samps = 10
    value = dsp.calc_num_overlap_samples(
        samples_per_frame=20, percent_overlap=50)
    assert value == overlap_samps


def test_calc_num_subframes():
    num_subframes = 5
    value = dsp.calc_num_subframes(
        tot_samples=30, frame_length=10, overlap_samples=5)
    assert value == num_subframes


def test_create_window_hamming():
    window_samples = np.array([0.08, 0.54, 1., 0.54, 0.08])
    value_array = dsp.create_window(window_type='hamming',
                                    frame_length=5)
    assert np.allclose(value_array, window_samples)


def test_create_window_hann():
    window_samples = np.array([0., 0.5, 1., 0.5, 0.])
    value_array = dsp.create_window(window_type='hann',
                                    frame_length=5)
    assert np.allclose(value_array, window_samples)


def test_apply_window_hamming():
    input_signal = np.array([0.,  0.36371897, -0.302721,
                             -0.1117662,  0.3957433])
    window = np.array([0.08, 0.54, 1., 0.54, 0.08])
    windowed_signal = np.array([0.,  0.19640824, -0.302721,
                                -0.06035375,  0.03165946])
    value_array = dsp.apply_window(input_signal, window)
    assert np.allclose(value_array, windowed_signal)


def test_apply_window_hann():
    input_signal = np.array([0.,  0.36371897, -0.302721,
                             -0.1117662,  0.3957433])
    window = np.array([0., 0.5, 1., 0.5, 0.])
    windowed_signal = np.array([0.,  0.18185948, -0.302721,
                                -0.0558831,  0.])
    value_array = dsp.apply_window(input_signal, window)
    assert np.allclose(value_array, windowed_signal)


def test_apply_window_too_small_window():
    input_signal = np.array([0.,  0.36371897, -0.302721,
                             -0.1117662,  0.3957433])
    window_small = np.array([0., 0.75, 0.75, 0.])
    with pytest.raises(ValueError):
        dsp.apply_window(input_signal, window_small)


def test_apply_window_too_big_window():
    input_signal = np.array([0.,  0.36371897, -0.302721,
                             -0.1117662,  0.3957433])
    window_big = np.array([0., 0.3454915, 0.9045085, 0.9045085,
                           0.3454915, 0.])
    with pytest.raises(ValueError):
        dsp.apply_window(input_signal, window_big)


def test_calc_power_real():
    power = np.array([0., 1., 4., 4.])
    input_array = np.array([0, -2, -4, 4, ])
    value_array = dsp.calc_power(input_array)
    assert np.allclose(value_array, power)


def test_calc_power_complex():
    power = np.array([[0.33333333, 0.33333333, 0.33333333],
                      [1.33333333, 1.33333333, 1.33333333],
                      [3., 3., 3.]])
    input_array = np.array(
        [[1, 1, 1], [2j, 2j, 2j], [-3, -3, -3]], dtype=np.complex_)
    value_array = dsp.calc_power(input_array)
    assert np.allclose(value_array, power)


def test_calc_average_power():
    average_power = np.array([[2., 2., 2.],
                              [1., 1., 1.],
                              [0.33333333, 0.33333333, 0.33333333]])
    input_array = np.array([[6, 6, 6], [3, 3, 3], [1, 1, 1]])
    value_array = dsp.calc_average_power(input_array, num_iters=3)
    assert np.allclose(value_array, average_power)


def test_calc_posteri_snr():
    snr = np.array([3., 3., 3., 3.])
    sig_power = np.array([6, 6, 6, 6])
    noise_power = np.array([2, 2, 2, 2])
    value_array = dsp.calc_posteri_snr(sig_power, noise_power)
    assert np.allclose(value_array, snr)


def test_calc_posteri_prime():
    snr = np.array([3., 2., 1., 0.])
    snr_prime = np.array([2., 1., 0., 0.])
    value_array = dsp.calc_posteri_prime(snr)
    assert np.allclose(value_array, snr_prime)


def test_calc_prior_snr_first_iter_None():
    prior_snr = np.array([2.98, 1.98, 0.98, 0.])
    snr = np.array([3., 2., 1., 0.])
    snr_prime = np.array([2., 1., 0., 0.])
    smooth_factor = 0.98
    first_iter = None
    gain = None
    value_array = dsp.calc_prior_snr(
        snr, snr_prime, smooth_factor, first_iter, gain)
    assert np.allclose(value_array, prior_snr)


def test_calc_prior_snr_first_iter_True():
    prior_snr = np.array([1.02, 1., 0.98, 0.98])
    snr = np.array([3., 2., 1., 0.])
    snr_prime = np.array([2., 1., 0., 0.])
    smooth_factor = 0.98
    first_iter = True
    gain = None
    value_array = dsp.calc_prior_snr(
        snr, snr_prime, smooth_factor, first_iter, gain)
    assert np.allclose(value_array, prior_snr)


def test_calc_prior_snr_first_iter_False_gain():
    prior_snr = np.array([0.775, 0.51, 0.245, 0.])
    snr = np.array([3., 2., 1., 0.])
    snr_prime = np.array([2., 1., 0., 0.])
    smooth_factor = 0.98
    first_iter = False
    gain = np.array([0.5, 0.5, 0.5, 0.5])
    value_array = dsp.calc_prior_snr(
        snr, snr_prime, smooth_factor, first_iter, gain)
    assert np.allclose(value_array, prior_snr)


def test_calc_gain_power_estimation():
    gain = np.array([0.71059869, 0.70710678, 0.70352647, 0.70352647])
    prior_snr = np.array([1.02, 1., 0.98, 0.98])
    value_array = dsp.calc_gain(prior_snr)
    assert np.allclose(value_array, gain)


def test_apply_gain_fft():
    fft_gain = np.array([0.99525999+0.j, 0.9952708 - 0.00234968j,
                         0.99530319-0.0046995j, 0.99535717-0.00704959j])
    fft_array = np.array([1.99051999+0.j, 1.99054159-0.00469936j,
                          1.99060637-0.009399j, 1.99071435-0.01409919j])
    gain = np.array([0.5, 0.5, 0.5, 0.5])
    value_array = dsp.apply_gain_fft(fft_array, gain)
    assert np.allclose(fft_gain, value_array)


def test_calc_ifft():
    ifft_array = np.array([1.99059557e+00-0.00704939j, -2.37155250e-03+0.00230656j,
                           -3.23950000e-05+0.00234989j, 2.32836250e-03+0.00239294j])
    fft_array = np.array([1.99051999+0.j, 1.99054159-0.00469936j,
                          1.99060637-0.009399j, 1.99071435-0.01409919j])
    value_array = dsp.calc_ifft(fft_array)
    assert np.allclose(ifft_array, value_array)


def test_control_volume_lower():
    vol_controlled = np.array([0.32, -0.13316698, -0.20916596,
                               0.30725449, -0.04656001])
    sample_array = np.array([0.4, -0.16645873, -0.26145745,  0.38406811,
                             -0.05820001])
    value_array = dsp.control_volume(
        sample_array, min_limit=0.12, max_limit=0.35)
    assert np.allclose(vol_controlled, value_array)


def test_control_volume_increase():
    vol_controlled = np.array([0.78125, -0.32511471, -0.51065908,
                               0.75013303, -0.11367189])
    sample_array = np.array([0.4, -0.16645873, -0.26145745,  0.38406811,
                             -0.05820001])
    value_array = dsp.control_volume(
        sample_array, min_limit=0.72, max_limit=0.85)
    assert np.allclose(vol_controlled, value_array)


def test_add_tensor():
    output = np.array([[[[0], [1]], [[2], [3]], [[4], [5]]],
                       [[[6], [7]], [[8], [9]], [[10], [11]]]])
    matrix = np.arange(12).reshape((2, 3, 2))
    value_array = matrixfun.add_tensor(matrix)
    assert np.array_equal(output, value_array)


def test_add_tensor_valueerror_empty_input():
    with pytest.raises(ValueError):
        value_array = matrixfun.add_tensor(np.array([]))


def test_add_tensor_typeerror_list():
    matrix = np.arange(12).reshape((2, 3, 2))
    with pytest.raises(TypeError):
        value_array = matrixfun.add_tensor(list(matrix))


def test_create_empty_matrix_floats():
    output = np.array([[0., 0., 0., 0.],
                       [0., 0., 0., 0.],
                       [0., 0., 0., 0.]])
    value_array = matrixfun.create_empty_matrix((3, 4))
    assert np.array_equal(output, value_array)


def test_create_empty_matrix_complex():
    output = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
    value_array = matrixfun.create_empty_matrix((3, 4), complex_vals=True)
    assert np.array_equal(output, value_array)


def test_create_empty_matrix_empty_float():
    output = np.array([], dtype=np.complex128)
    value_array = matrixfun.create_empty_matrix(0)
    assert np.array_equal(output, value_array)


def test_create_empty_matrix_empty_complex():
    output = np.array([], dtype=np.float64)
    value_array = matrixfun.create_empty_matrix(0, complex_vals=True)
    assert np.array_equal(output, value_array)


def test_create_empty_matrix_vector_float():
    output = np.array([0., 0., 0., 0., 0.])
    value_array = matrixfun.create_empty_matrix(5)
    assert np.array_equal(output, value_array)


def test_create_empty_matrix_vector_complex():
    output = np.array([0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])
    value_array = matrixfun.create_empty_matrix(5, complex_vals=True)
    assert np.array_equal(output, value_array)


def test_separate_dependent_var_vector():
    ind_vals = np.array([1, 2, 3])
    dep_val = 4
    value_array, value = matrixfun.separate_dependent_var(
        np.array([1, 2, 3, 4]))
    assert np.array_equal(ind_vals, value_array)
    assert dep_val == value


def test_separate_dependent_var_matrix():
    matrix = np.arange(20).reshape((2, 2, 5))
    ind_vals = np.array([[[0,  1,  2,  3], [5,  6,  7,  8]],
                         [[10, 11, 12, 13], [15, 16, 17, 18]]])
    dep_vals = np.array([4, 14])
    value_array1, value_array2 = matrixfun.separate_dependent_var(matrix)
    assert np.array_equal(ind_vals, value_array1)
    assert np.array_equal(dep_vals, value_array2)


def test_separate_dependent_var_indexerror_empty_input_array():
    with pytest.raises(IndexError):
        matrixfun.separate_dependent_var(np.array([]))


def test_collect_features_shorter_signal_than_window_size():
    input_sig = np.array([0., 0.00999896, 0.01999167, 0.02997188,
                          0.03993337, 0.04986989, 0.05977525, 0.06964326])
    input_window_size = 20
    mfcc_vals = np.array([[-7.15553843, -28.42296241,  -4.63982511, -14.99670075,
                           2.68488118,  45.81883864,  28.47780657, -65.34741205,
                           -77.81374514,  86.96464875,  81.20218644, -79.04416609,
                           -10.01916891]])
    frame_length = 8
    adjusted_window_size = 1.0
    value_array, value1, value2 = dsp.collect_features(
        samples=input_sig,
        feature_type='mfcc',
        sr=8000,
        window_size_ms=input_window_size,
        window_shift_ms=input_window_size/2.,
        num_mfcc=13
    )
    assert np.allclose(mfcc_vals, value_array)
    assert frame_length == value1
    assert adjusted_window_size == value2
    assert input_window_size > value2
