#!/usr/bin/python
"""
Simulate fMRI run with no activation.
"""

# parse as command line argument
# bids_dir


def simulate_run(infile, outfile, lfnl=3.0, hfnl=None):

    # TODO: make a workflow out of it
    from nipype.interfaces import fsl

    # perform motion correction using mcflirt implemented by nipype
    mcfile = infile.replace('.nii.gz', '_mc.nii.gz')
    mcflt = fsl.MCFLIRT(in_file=infile,
                        out_file=mcfile)
    mcflt.run()

    # load the preprocessed nifti as pymvpa data set
    from mvpa2.datasets.mri import fmri_dataset, map2nifti
    ds = fmri_dataset(mcfile)

    # get TR from sample attributes
    tr = ds.sa['time_coords'][1] - ds.sa['time_coords'][0]
    # convert to sampling rate in Hz
    sr = 1 / tr
    cutoff = 1 / (2 * tr)    # TODO: what would be a good LPF cutoff?
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


def run4bids(bids_dir):

    import os

    subdirs = [directory for directory in os.listdir(bids_dir) if directory.startswith('sub-')]

    for sub in subdirs:

        funcs = [dircontent for dircontent in os.listdir(os.path.join(bids_dir, sub, 'func'))
                 if dircontent.endswith('.nii.gz')]

        for func in funcs:
            in_file = os.path.join(bids_dir, sub, 'func', func)
            out_file = os.path.join(bids_dir, 'derivatives', sub, 'shambrain',
                                    func.replace('.nii.gz', '_sham.nii.gz'))
            simulate_run(in_file, out_file)
