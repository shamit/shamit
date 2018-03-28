#!/usr/bin/python

"""
Run simulation over our data
"""
__author__ = "Oliver Contier"


if __name__ == "__main__":

    from os.path import join
    import copy
    from famface_simulation_functions import *
    from mvpa2.datasets.mri import fmri_dataset, map2nifti
    import sys

    sub = sys.argv[1]
    data_basedir = sys.argv[2]
    maskpath = sys.argv[3]
    workdirbase = sys.argv[4]

    # template for directory name containing subjects data in the working directory
    subdir_template = '_model_id_1_subject_id_%s_task_id_1' % sub

    # path to anatomical image
    anat = join('/data/famface/openfmri/oli/results/extract_betas/l1_workdir_betas/',
                'registration', subdir_template, 'stripper',
                'highres001_brain.nii.gz')

    # path to hd5 file and affine transformation matrix
    mni2anat_hd5 = join('/data/famface/openfmri/oli/results/extract_betas/l1_workdir_betas/',
                        'registration', subdir_template,
                        'antsRegister', 'output_InverseComposite.h5')
    affine_matrix = join('/data/famface/openfmri/oli/results/extract_betas/l1_workdir_betas/',
                         'registration', subdir_template,
                         'mean2anatbbr', 'median_flirt.mat')

    # maskpath = '/data/famface/openfmri/scripts/notebooks/rois_manual_r5_20170222_nooverlap.nii.gz'

    # initial amplitudes for fam and unfam (first run). will be changed for each new run.
    amplitudes = [8, 1]

    # get runs
    runs_unsorted = os.listdir(os.path.join(data_basedir, sub, 'model/model001/onsets'))
    runs = [run for run in sorted(runs_unsorted)]

    for run in runs:

        # create run specific working directory
        workdir = os.path.join(workdirbase, sub, run)
        if not os.path.exists(workdir):
            os.makedirs(workdir)

        # path to bold file
        boldfile = join(data_basedir, sub, 'BOLD', run, 'bold.nii.gz')

        # simulate noise image
        noise = simulate_run(boldfile, workdir)

        # get onsets
        spec = get_onsets_famface(
            os.path.join(data_basedir, sub, 'model/model001/onsets', run),
            amplitudes)

        # increment amplitude for unfam
        amplitudes[1] += 0.5

        # create second specification for contrast that does not increase
        # in different roi
        straightspec = copy.deepcopy(spec)
        straightspec[0]['amplitude'] = 8
        straightspec[1]['amplitude'] = 1
        for cond in straightspec:
            cond['roivalue'] = 24
            spec.append(cond)

        # transform mask to subject space
        mask_subjspace = mask2subjspace_real(maskpath, anat, boldfile,mni2anat_hd5, affine_matrix, workdir)

        # add signal
        with_signal = add_signal_custom(noise, mask_subjspace, spec)

        # save data
        image = map2nifti(with_signal)
        image.to_filename(
            os.path.join(data_basedir, sub, 'BOLD', run, 'sim.nii.gz'))
