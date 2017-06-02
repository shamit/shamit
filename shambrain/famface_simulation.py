#!/usr/bin/python

"""
Run simulation over our data
"""

if __name__ == "__main__":

    import os
    from shambrain import *
    from mvpa2.datasets.mri import fmri_dataset, map2nifti

    amplitudes = [8, 1]
    runs = os.listdir('/data/famface/openfmri/results/l1ants_final/model001/task001/sub001/bold/')

    # get mask
    ms = fmri_dataset('/data/famface/openfmri/scripts/notebooks/'
                      'rois_manual_r5_20170222_nooverlap.nii.gz')

    for run in runs:

        workdir = os.path.join('/data/famface/openfmri/oli/simulation/mcfiles', run)
        if not os.path.exists(workdir):
            os.makedirs(workdir)

        prep_run_dir = run.replace('task001_', '')
        noise = simulate_run(
            os.path.join('/data/famface/openfmri/results/l1ants_final/model001/task001/sub001/bold/',
                         prep_run_dir, 'bold_mni.nii.gz'),
            workdir)

        # get onsets
        spec = get_onsets_famface(
            os.path.join('/data/famface/openfmri/oli/simulation/dataladclone'
                         '/sub002/model/model001/onsets', run),
            amplitudes)
        # add signal
        with_signal = add_signal_custom(noise, ms, spec)

        # save data
        image = map2nifti(with_signal)
        image.to_filename(
            os.path.join('/data/famface/openfmri/oli/simulation/dataladclone/'
                         'sub002/BOLD/', run, 'sim.nii.gz'))

    # get sub-id and run-number from command line arguments
    # should be: python famface_simulation.py sub001
    # import sys
    # sub = sys.argv[1]
    # run_number = sys.argv[2]
