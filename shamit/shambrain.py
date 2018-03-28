#!/usr/bin/python
# coding=utf-8
"""
fmri data simulation
"""
__author__ = "Oliver Contier"

import numpy as np
from mvpa2.datasets.mri import fmri_dataset

# TODO: problem is, BRAINIAK demands python 3.4
from brainiak.utils.fmrisim import generate_stimfunction, double_gamma_hrf

from mvpa2.misc.data_generators import autocorrelated_noise, simple_hrf_dataset
import csv
from nipype.interfaces import fsl
from nipype.interfaces.fsl.utils import ConvertXFM
from os.path import join
import os

"""

General procedure:

- get conditions
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
    """
    Perform motion correction on a dataset and load the result in pymvpa
    """
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


def mni2bold(bold, anat, standard,  mask, workdir):
    """
    Use fsl.FLIRT to transform a mask from MNI to subject space
    """

    os.makedirs(workdir)

    # bold to anat
    bold2anat = fsl.FLIRT(
        dof=6, no_clamp=True,
        in_file=bold,
        reference=anat,
        out_matrix_file=join(workdir, 'bold2anat.txt'),
        out_file=join(workdir, 'bold2anat.nii.gz'))
    bold2anat.run()

    # anat to mni
    anat2mni = fsl.FLIRT(
        dof=12,
        in_file=anat,
        reference=standard,
        out_matrix_file=join(workdir, 'anat2mni.txt'),
        out_file=join(workdir, sub, run, '%s_%s_anat2mni.nii.gz' % (sub, run)))
    anat2mni.run()

    # concatinate matrices
    concat = ConvertXFM(
        concat_xfm=True,
        in_file2=join(workdir, sub, run, '%s_%s_bold2anat.txt' % (sub, run)),
        in_file=join(workdir, sub, run, '%s_%s_anat2mni.txt' % (sub, run)),
        out_file=join(workdir, sub, run, '%s_%s_bold2mni.txt' % (sub, run)))
    concat.run()

    # inverse transmatrix
    inverse = ConvertXFM(
        in_file=join(workdir, sub, run, '%s_%s_bold2mni.txt' % (sub, run)),
        out_file=join(workdir, sub, run, '%s_%s_mni2bold.txt' % (sub, run)),
        invert_xfm=True)
    inverse.run()

    # apply to mask
    mni2bold = fsl.FLIRT(
        interp='nearestneighbour',
        apply_xfm=True,
        in_matrix_file=join(workdir, sub, run, '%s_%s_mni2bold.txt' % (sub, run)),
        in_file=join(mask),
        reference=join(data_basedir, sub, 'BOLD', run, 'bold.nii.gz'),
        out_file=join(workdir, sub, run, '%s_%s_roimask.nii.gz' % (sub, run)))
    mni2bold.run()

    mask_subjspace = join(workdir, sub, run, '%s_%s_roimask.nii.gz' % (sub, run))

    # return path of created mask
    return mask_subjspace


def get_conditions_openfmri(ds, model_id):
    """
    get names of conditions from participants_key.txt
    along with corresponding filenames for the onsets.
    """
    conditions_key_file = os.path.join(
        ds.basedir, 'models', 'model%03d' % (model_id), 'participants_key.txt')

    with open(conditions_key_file) as f:
        reader = csv.reader(f, delimiter='\t')
        condfiles = list(zip(*reader))[1]
        condnames = list(zip(*reader))[2]

    return condfiles, condnames


def get_events_info_openfmri(ds, model_id, task_id, sub, run,
                             condfiles, condnames):
    """
    for given run of given subject in an openfmri dataset, get the onsets and durations.
    get_conditions_openfmri should be run before.
    """
    events_dir = os.path.join(ds.basedir, sub, 'model', 'model%03d' % model_id,
                              'onsets', 'task%03d_run%03d' % (task_id, run), 'onset')  # /'condXX.txt'

    if set(condfiles) != set(sorted(os.listdir(events_dir))):
        raise ValueError('the list of conditions in participants_key.txt does not match'
                         'the conditions specified in the onsets directories.')
    else:

        design_spec = []

        for cf, cn in zip(condfiles, condnames):
            conddict = {}
            conddict['trial_type'] = cn
            with open(os.path.join(events_dir, '%s.txt' % cf)) as f:
                reader = csv.reader(f, delimiter='\t')
                conddict['onsets'] = list(zip(*reader))[0]
                conddict['durations'] = list(zip(*reader))[1]
            design_spec.append(conddict)

        return design_spec


def get_conditions_bids(infile):
    """
    read onsets, duration and amplitude from 3 column format file.
    (e.g. events.tsv in BIDS format)
    """

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


def get_signal_spec(infile, spec):
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
                    event['signal_type'] = row[header.index('signal_type')].split(',')
                else:
                    continue
    return spec


def univ_neural_signal(spec, tr, nvolumes):
    """
    For each event/condition in the input spec,
    generate a neural signal function (e.g. boxcar)
    and append it to the spec
    """

    for event in spec:
        # TODO: add option for lag
        if event['signal_type'] == 'boxcar':
            event['neural_signal'] = generate_stimfunction(event['onsets'],
                                                           event['durations'],
                                                           (nvolumes * tr),
                                                           weights=event['amplitudes'])

    return spec

    # TODO: More kinds of univariate signals


def neural_signal2bold(spec, tr):
    """
    For each event in spec, convolute the neural signal function with the specified
    convolutional model and append it to spec as 'bold_model'.
    """

    for event in spec:
        if event['convolution_model'] == 'double_gamma':
            event['bold_model'] = double_gamma_hrf(neural_signal, tr)

    return spec

    # TODO: more convolutional models


def make_noise(bold_dataset, tr, noisetype='autocorrelated',
               lfnl=3, hfnl=.5, seedval=1):
    """
    Simulate a functional image containing only noise using an existing
    image as a pedestal.
    """

    seed(seedval)

    if noisetype == 'autocorrelated':
        # convert to sampling rate in Hz
        sr = 1.0 / tr
        cutoff = sr / 4

        with_noise = autocorrelated_noise(bold_dataset, sr, cutoff, lfnl=lfnl, hfnl=hfnl)
        return with_noise

        # TODO: more types of noise


def add_signal(ds, ms, spec):
    """
    Basically the same as with add_signal_custom but onsets are shifted by lag.
    """

    # TODO: project mask into subject space!

    with_signal = ds.copy()

    for cond in spec:
        roivalue = cond['roivalue']
        bold_model = cond['bold_model']

        # get voxel indices for roi
        roi_indices = np.where(ms.samples[0] == roivalue)[0]

        # add bold activation model to noise
        for sample, activation in zip(with_signal.samples, bold_model):
            sample[roi_indices] += activation

        # TODO: add option for multiplicative vs. additive noise

    return with_signal
