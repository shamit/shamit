#!/usr/bin/python
# coding=utf-8
"""
fmri data simulation
"""

import numpy as np
from mvpa2.datasets.mri import fmri_dataset
from mvpa2.misc.data_generators import autocorrelated_noise
from mvpa2.misc.data_generators import simple_hrf_dataset
from nipype.interfaces import fsl
import csv
import os
import itertools


"""
General procedure:
- get design parameters
    for each expected effect / condition...
    . onsets
    . durations
    . amplitudes
    . roi(s)
    . lag
- Model Specifications:
    . within- & between run effect over time
    . type of neural signal model
    . which convolution
    . noise specifications
- make neueral signal model (e.g. boxcar)
- convolve to bold model (e.g. with double gamma)
- load volume and perform motion correction
- apply signal to volume
- generate noise
- add noise to volume
- output nifti image
"""


def load_and_mc(infile, workdir):
    # perform motion correction using mcflirt implemented by nipype.
    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

    infilename = infile.split('/')[-1]
    mcfile = os.path.join(workdir, infilename.replace('.nii.gz', '_mc.nii.gz'))

    mcflt = fsl.MCFLIRT(in_file=infile,
                        out_file=mcfile)
    mcflt.run()

    # load the preprocessed nifti as pymvpa data set
    dataset = fmri_dataset(mcfile)
    tr = float(dataset.sa['time_coords'][1] - dataset.sa['time_coords'][0])
    nvolumes = len(dataset.samples)

    return dataset, tr, nvolumes


def get_events_info(infile):
    """
    read onsets, duration and amplitude from 3 column format file.
    (e.g. events.tsv in BIDS format)
    """

    # TODO: make it work with non-tsv input files.

    spec = []

    if infile.endswith('.tsv'):

        # read tsv file
        with open(infile) as f:
            reader = csv.reader(f, delimiter='\t')
            rows = [row for row in reader]
            header = rows[0]
            trial_types = set([row[header.index('trial_type')] for row in rows])

        # create a dict for each trial type
        for event in trial_types:
            event_dict = {
                'trial_type': event,
                'onsets': [float(row[header.index('onset')]) for row in rows[1:]
                           if row[header.index('trial_type') == event]],
                'durations': [float(row[header.index('duration')]) for row in rows[1:]
                             if row[header.index('trial_type') == event]]
            }
            spec.append(event_dict)
    return spec


def get_design_spec(infile, spec):

    """
    add info about amplitude, rois, changes in amplitude over time, and lag to spec
    (for each event / trial_type)
    """
    with open(infile) as f:
        reader = csv.reader(f, delimiter='\t')
        rows = [row for row in reader if not row[0].startswith('#')]
        header = rows[0]
        trial_types = set([row[header.index('trial_type')] for row in rows])

    for trial_type in trial_types:
        for event in spec:
            for row in rows:
                if event['trial_type'] == trial_type:
                    event['amplitude'] = row[header.index('amplitude')]
                    event['lag'] = row[header.index('lag')]
                    event['sample_scalar'] = row[header.index('sample_scalar')]
                    event['run_scalar'] = row[header.index('run_scalar')]
                    event['rois'] = row[header.index('rois')].split(',')
                else:
                    continue
    return spec


def univ_neural_signal(tr, nvolumes, onsets, durations, amplitudes,
                       function='boxcar'):

    if function == 'boxcar':
        from fmrisim import generate_stimfunction
        boxcar = generate_stimfunction(onsets, durations, (nvolumes*tr), weights=amplitudes)

        return boxcar

    # TODO: More kinds of univariate signals


def neural_to_bold(neural_signal, tr, model='double_gamma'):

    if model == 'double_gamma':

        from fmrisim import double_gamma_hrf
        bold_model = double_gamma_hrf(neural_signal, tr)

        return bold_model

    # TODO: more convolutional models


def make_noise(dataset, tr, noisetype='autocorrelated',
               lfnl=3, hfnl=.5):

    if noisetype == 'autocorrelated':

        # convert to sampling rate in Hz
        sr = 1.0 / tr
        cutoff = sr / 4

        with_noise = autocorrelated_noise(dataset, sr, cutoff, lfnl=lfnl, hfnl=hfnl)
        return with_noise

    # TODO: more types of noise


def add_signal_lagged(ds, ms, spec):
    """
    Basically the same as with add_signal_custom but onsets are shifted by lag.
    """

    with_lagged_signal = ds.copy()
    tr = ds.sa['time_coords'][1] - ds.sa['time_coords'][0]
    nsamples = len(ds.samples)

    """
    Get parameters from spec
    """
    for cond in spec:
        # condition = spec['conditions'][cond]
        roivalue = cond['roivalue']
        amplitude = cond['amplitude']
        sigchange = float(amplitude) / 100

        """
        shift onsets by lag in TR units
        """
        lag = cond['lag']
        lag_tr = lag * tr
        onsets = [ons + lag_tr for ons in cond['onset']]

        # get voxel indices for roi
        roi_indices = np.where(ms.samples[0] == roivalue)[0]

        """
        model hrf
        """
        hrf_model = simple_hrf_dataset(events=onsets, nsamples=nsamples * 2, tr=tr, tres=1, baseline=1,
                                       signal_level=sigchange,
                                       noise_level=0).samples[:, 0]
        """
        add activation to data set
        """
        for sample, activation in zip(with_lagged_signal.samples, hrf_model):
            sample[roi_indices] *= activation

    return with_lagged_signal
