#!/usr/bin/python
"""
Simulate fMRI run with no activation.
"""

# parse as command line argument
# bids_dir


def simulate_run(infile, outfile, lfnl=3.0, hfnl=None):
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

    from nipype.interfaces import fsl
    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

    # perform motion correction using mcflirt implemented by nipype.
    mcfile = infile.replace('.nii.gz', '_mc.nii.gz')
    mcflt = fsl.MCFLIRT(in_file=infile,
                        out_file=mcfile)
    mcflt.run()

    # load the preprocessed nifti as pymvpa data set
    from mvpa2.datasets.mri import fmri_dataset, map2nifti
    ds = fmri_dataset(mcfile)

    # get TR from sample attributes
    tr = float(ds.sa['time_coords'][1] - ds.sa['time_coords'][0])

    # convert to sampling rate in Hz
    sr = 1.0 / tr
    cutoff = 1.0 / (2.0 * tr)

    # produce simulated 4D fmri data
    # mandatory inputs are dataset, sampling rate (Hz),
    # and cutoff frequency of the low-pass filter.
    # optional inputs are low frequency noise (%) (lfnl)
    # and high frequency noise (%) (hfnl)

    from mvpa2.misc.data_generators import autocorrelated_noise
    shambold = autocorrelated_noise(ds, sr, cutoff, lfnl, hfnl)

    # save to nifti file
    image = map2nifti(shambold)
    image.to_filename(outfile)


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
    import os

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
