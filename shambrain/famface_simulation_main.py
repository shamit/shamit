#!/usr/bin/python

"""
Run simulation over our data
"""

if __name__ == "__main__":

    import os
    from famface_simulation_functinos import *
    from mvpa2.datasets.mri import fmri_dataset, map2nifti
    import sys

    sub = sys.argv[1]

    amplitudes = [8, 1]
    runs_unsorted = os.listdir(
        os.path.join('/data/famface/openfmri/data/',
                     sub, 'model/model001/onsets'))
    runs = [run for run in sorted(runs_unsorted)]

    # get mask
    ms = fmri_dataset('/data/famface/openfmri/scripts/notebooks/'
                      'rois_manual_r5_20170222_nooverlap.nii.gz')

    for run in runs:

        workdir = os.path.join('/data/famface/openfmri/oli/simulation/mcfiles', sub, run)
        if not os.path.exists(workdir):
            os.makedirs(workdir)

        mni_run_dir = run.replace('task001_', '')
        noise = simulate_run(
            os.path.join('/data/famface/openfmri/results/l1ants_final/model001/task001/',
                         sub, 'bold/', mni_run_dir, 'bold_mni.nii.gz'),
            workdir)

        # get onsets
        spec = get_onsets_famface(
            os.path.join('/data/famface/openfmri/data/',
                         sub, 'model/model001/onsets', run),
            amplitudes)
        amplitudes[1] += 0.5

        # create second specification for contrast that does not increase and give it another roi value
        import copy
        straightspec = copy.deepcopy(spec)
        for cond in straightspec:
            cond['amplitude'] = 8
            cond['roivalue'] = 24
            spec.append(cond)

        # add signal
        with_signal = add_signal_custom(noise, ms, spec)

        # save data
        image = map2nifti(with_signal)
        image.to_filename(
            os.path.join('/data/famface/openfmri/data/',
                         sub, 'BOLD', run, 'sim.nii.gz'))
