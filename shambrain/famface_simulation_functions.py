#!/usr/bin/python
# coding=utf-8
"""
Simulate fMRI run with no activation.
"""

import numpy as np
from mvpa2.datasets.mri import fmri_dataset
from mvpa2.misc.data_generators import autocorrelated_noise
from mvpa2.misc.data_generators import simple_hrf_dataset
from nipype.interfaces import fsl
import csv
import os
from os.path import join
import itertools
from nipype.interfaces import fsl
from nipype.interfaces.fsl.utils import ConvertXFM


def simulate_run(infile, workdir, lfnl=3, hfnl=.5):
    """
    Simple simulation of 4D fmri data. Takes a given functional image,
    performs motion correction, computes the mean and adds autocorrelated
    noise to it.

    Parameters
    ----------
    infile:     str
        Path to the 4D functional image in .nifti format
    outfile:    str
        Path and name of the simulated image
    lfnl:       float
        Low frequency noise level. Default = 3 Hz
    hfnl:       float
        High frequency noise level. Default = None.
    """

    # perform motion correction using mcflirt implemented by nipype.
    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
    infilename = infile.split('/')[-1]
    mcfile = os.path.join(workdir, infilename.replace('.nii.gz', '_mc.nii.gz'))
    mcflt = fsl.MCFLIRT(in_file=infile,
                        out_file=mcfile)
    mcflt.run()

    # load the preprocessed nifti as pymvpa data set
    ds = fmri_dataset(mcfile)

    # get TR from sample attributes
    tr = float(ds.sa['time_coords'][1] - ds.sa['time_coords'][0])

    # convert to sampling rate in Hz
    sr = 1.0 / tr
    cutoff = sr / 4

    # produce simulated 4D fmri data
    shambold = autocorrelated_noise(ds, sr, cutoff, lfnl=lfnl, hfnl=hfnl)
    return shambold


def get_onsets_famface(inpath, amplitudes):
    """
    get event onsets from text file
    """

    if not os.path.exists(inpath):
        raise ValueError('the specified input path does not exist')
    else:
        confiles = os.listdir(inpath)

    famfiles = ['cond006.txt', 'cond007.txt', 'cond008.txt', 'cond009.txt']
    fam_paths = [os.path.join(inpath, confile) for confile in confiles if confile in famfiles]

    unfamfiles = ['cond002.txt', 'cond003.txt', 'cond004.txt', 'cond005.txt']
    unfam_paths = [os.path.join(inpath, confile) for confile in confiles if confile in unfamfiles]

    # collect onsets of all familiar faces
    fam_onsets = []
    for fam in fam_paths:
        with open(fam) as f:
            reader = csv.reader(f, delimiter='\t')
            fam_onsets.append(list(zip(*reader))[0])
    fam_onsets = list(itertools.chain(*fam_onsets))
    fam_onsets = sorted([float(i) for i in fam_onsets])

    # and for unfamiliar faces
    unfam_onsets = []
    for unfam in unfam_paths:
        with open(unfam) as f:
            reader = csv.reader(f, delimiter='\t')
            unfam_onsets.append(list(zip(*reader))[0])
    unfam_onsets = list(itertools.chain(*unfam_onsets))
    unfam_onsets = sorted([float(i) for i in unfam_onsets])

    spec = [{'chunks': 0, 'duration': 1.5, 'onset': fam_onsets, 'targets': 'familiar', 'amplitude': amplitudes[0], 'roivalue': 2},
            {'chunks': 0, 'duration': 1.5, 'onset': unfam_onsets, 'targets': 'unfamiliar', 'amplitude': amplitudes[1],
             'roivalue': 2}]

    return spec


def add_signal_custom(ds, ms, spec, tpeak=0.8, fwhm=1, fir_length=15):
    """
    add signal to a pure noise simulated image
    (as generated e.g. by simulate_run)
    """

    dataset_with_signal = ds.copy()
    ms = fmri_dataset(ms)

    """
    some parameters from data
    """
    #  TR
    tr = ds.sa['time_coords'][1] - ds.sa['time_coords'][0]
    nsamples = len(ds.samples)

    """
    loop over specified conditions
    """
    for cond in spec:
        # condition = spec['conditions'][cond]
        roivalue = cond['roivalue']
        onsets = cond['onset']
        amplitude = cond['amplitude']
        sigchange = float(amplitude) / 100

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
        # add model activation to roi voxels
        # import pdb; pdb.set_trace()
        for sample, activation in zip(dataset_with_signal.samples, hrf_model):
            sample[roi_indices] *= activation

    return dataset_with_signal


def mask2subjspace(sub, run, data_basedir, workdir, mask):
    """
    Use fsl.FLIRT to transform roi mask from MNI to subject space (for each run)
    """
    from nipype.interfaces import fsl
    from nipype.interfaces.fsl.utils import ConvertXFM
    from os.path import join
    import os

    os.makedirs(join(workdir, sub, run))

    # bold to anat
    bold2anat = fsl.FLIRT(
        dof=6, no_clamp=True,
        in_file=join(data_basedir, sub, 'BOLD', run, 'bold.nii.gz'),
        reference=join(data_basedir, sub, 'anatomy', 'highres001.nii.gz'),
        out_matrix_file=join(workdir, sub, run, '%s_%s_bold2anat.txt' % (sub, run)),
        out_file=join(workdir, sub, run, '%s_%s_bold2anat.nii.gz' % (sub, run)))
    bold2anat.run()

    # anat to mni
    anat2mni = fsl.FLIRT(
        dof=12, interp='nearestneighbour',
        in_file=join(data_basedir, sub, 'anatomy', 'highres001.nii.gz'),
        reference='/usr/share/fsl/5.0//data/standard/MNI152_T1_2mm_brain.nii.gz',
        out_matrix_file=join(workdir, sub, run, '%s_%s_anat2mni.txt' % (sub, run)),
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
    return mask_subjspace


def get_filepaths_bids(bids_dir):
    """
    Get the input and outputh pathnames given a BIDS dataset.

    Parameters
    ----------
    bids_dir:   str
        Base directory of the BIDS data set. Should contain the subject
        folders ('sub-001' etc.)

    Returns
    -------
    infiles:    list
        List of path names for all files in the subjects' 'func' subdirectory
    outfiles:   list
        List of path names for the to be saved simulated functinoal images.
        Will be saved under '/derivatives/sub-XX/sham/'.
    """

    infiles, outfiles = [], []

    subdirs = [directory for directory in os.listdir(bids_dir)
               if directory.startswith('sub-')]

    for sub in subdirs:

        funcs = [dircontent for dircontent in
                 os.listdir(os.path.join(bids_dir, sub, 'func'))
                 if dircontent.endswith('.nii.gz')]

        for func in funcs:
            infiles.append(os.path.join(bids_dir, sub, 'func', func))
            outfiles.append(os.path.join(
                bids_dir, 'derivatives', sub, 'shambrain',
                func.replace('.nii.gz', '_sham.nii.gz')))

    return infiles, outfiles


def run4bids(bids_dir):
    """
    runs simulation for all functional images in a BIDS data set.

    Parameters
    ----------
    bids_dir:   str
        Base directory of the BIDS data set. Should contain the subject
        folders ('sub-001' etc.)
    """

    for inf, outf in zip(get_filepaths_bids(bids_dir)):
        simulate_run(inf, outf)

# TODO: function to create a condor submission file
