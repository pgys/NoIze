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
'''Audio data is augmented to result in more resiliant models and filters.
'''

import numpy as np


def adjust_volume(samples, vol_range):
    samps = samples.copy()
    adjusted_volume = np.interp(samps,
                                (samps.min(), samps.max()),
                                (-vol_range, vol_range))
    return adjusted_volume

def spread_volumes(samples):
    '''returns samples with a range of volumes 
    '''
    if samples is None or len(samples) == 0:
        raise ValueError('No Audio sample data recognized.')
    max_vol = max(samples)
    if round(max_vol) > 1:
        raise ValueError('Audio data not normalized.')
    vol_range_high = adjust_volume(samples, 0.5)
    vol_range_mid = adjust_volume(samples, 0.3)
    vol_range_low = adjust_volume(samples, 0.1)
    for vol_set in (vol_range_high, vol_range_mid, vol_range_low):
        assert max(vol_set) < 1 and max(vol_set) > 0
    return vol_range_low, vol_range_mid, vol_range_high
