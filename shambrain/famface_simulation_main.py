#!/usr/bin/python

"""
Run simulation over our data
"""

if __name__ == "__main__":

    import os
    from famface_simulation_functions import *
    from mvpa2.datasets.mri import fmri_dataset, map2nifti
    import sys

    sub = sys.argv[1]
    data_basedir = sys.argv[2]
    maskpath = sys.argv[3]
    workdirbase = sys.argv[4]

    # data_basedidr = '/data/famface/openfmri/oli/simulation/data'
    # maskpath = '/data/famface/openfmri/scripts/notebooks/rois_manual_r5_20170222_nooverlap.nii.gz'

    amplitudes = [8, 1]
    runs_unsorted = os.listdir(os.path.join(data_basedir, sub, 'model/model001/onsets'))
    runs = [run for run in sorted(runs_unsorted)]

    for run in runs:

        workdir = os.path.join(workdirbase, sub, run)
        if not os.path.exists(workdir):
            os.makedirs(workdir)

        mni_run_dir = run.replace('task001_', '')

        # TODO: change input bold (subject space)
        noise = simulate_run(os.path.join(data_basedir, sub, 'BOLD', run, 'bold.nii.gz'), workdir)

        # get onsets
        spec = get_onsets_famface(
            os.path.join(data_basedir, sub, 'model/model001/onsets', run),
            amplitudes)
        amplitudes[1] += 0.5

        # create second specification for contrast that does not increase and give it another roi value
        import copy

        straightspec = copy.deepcopy(spec)
        straightspec[0]['amplitude'] = 8
        straightspec[1]['amplitude'] = 1
        for cond in straightspec:
            cond['roivalue'] = 24
            spec.append(cond)

        # TODO: mask_subjspace
        mask_subjspace = mask2subjspace(sub, run, data_basedir, workdir, maskpath)

        # add signal
        with_signal = add_signal_custom(noise, mask_subjspace, spec)

        # save data
        image = map2nifti(with_signal)
        image.to_filename(
            os.path.join(data_basedir, sub, 'BOLD', run, 'sim.nii.gz'))
