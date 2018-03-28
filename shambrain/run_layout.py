"""
Main run script for simulation workflow
"""
__author__ = "Oliver Contier"

import os
from mvpa2.datasets.sources.openfmri import OpenFMRIDataset
from shambrain import *


# tasks = ds.get_task_descriptions().keys()
# we could say 'for task in tasks:' here ...


def run_simulation_openfmri(dspath, roimask, signal_spec_file,
                            noisetype='autocorrelated',
                            workdir=os.path.join(dspath, 'shambrain_workdir'),
                            model_id=1, task_id=1,
                            seedval=1, lfnl=3, hfnl=.5):
    """
    Simulate a dataset on the basis of an existing dataset in OpenfMRI format.
    Makes use of OpenfMRI dataset functionalities implemented in PyMVPA2.
    """

    # get dataset layout in openfmri format
    ds = OpenFMRIDataset(dspath)

    # load roimask
    ms = fmri_dataset(roimask)

    # get conditions specified in the openfmri dataset
    condfiles, condnames = get_conditions_openfmri(ds, model_id)

    for subj in ds.get_subj_ids():
        for run in ds.get_bold_run_ids(subj, task_id):

            # perform motion correction and load result as pymvpa dataset
            bold_path = os.path.join(dspath, 'sub%03d' % sub, 'BOLD',
                                     'task%03d_run%03d' % (task_id, run),
                                     'bold.nii.gz')
            bold_dataset = load_and_mc(bold_path, workdir)

            # compute tr from data
            tr = float(bold_dataset.sa['time_coords'][1] - bold_dataset.sa['time_coords'][0])

            # get onsets and durations for each condition
            design_spec = get_events_info_openfmri(ds, model_id, task_id, sub, run,
                                                   condfiles, condnames)

            # get signal properties for simulation purposes as specified by the user
            # in a signal_spec_file. (e.g. amplitudes, type of signal_function,
            # change of amplitudes over time, ...)
            neural_signal_spec = get_signal_spec(signal_spec_file, design_spec)

            # convolve the neural signal with the convolutional model to get the BOLD signal.
            bold_signal_spec = neural_signal2bold(neural_signal_spec, tr)

            # Simulate Noise
            noise = make_noise(bold_dataset, tr, noisetype,
                               lfnl, hfnl, seedval)

            # TODO: project mask into subject space before doing this

            # Add signal
            with_signal = add_signal(noise, ms, bold_signal_spec)

            # save the simulated data
            image = map2nifti(with_signal)
            image.to_filename(os.path.join(dspath, 'sub%03d' % sub, 'BOLD',
                                           'task%03d_run%03d' % (task_id, run),
                                           'sim.nii.gz'))
