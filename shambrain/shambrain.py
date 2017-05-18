"""
Simulate fMRI run with no activation.
"""


def simulate_run(infile, outfile, sr, cutoff):

    # TODO: make a workflow out of it
    from nipype.interfaces import fsl

    # perform motion correction using mcflirt implemented by nipype
    mcfile = '{}_mc.nii.gz'.format(infile)
    mcflt = fsl.MCFLIRT(in_file=infile,
                        out_file=mcfile)
    mcflt.run()

    # load the preprocessed nifti as pymvpa data set
    from mvpa2.datasets.mri import fmri_dataset, map2nifti
    ds = fmri_dataset(mcfile)

    # produce simulated 4D fmri data
    # mandatory inputs are dataset, sampling rate (Hz),
    # and cutoff frequency of the low-pass filter.
    # optional inputs are low frequency noise (%) (lfnl)
    # and high frequency noise (%) (hfnl)
    sr = 0.5
    cutoff = 0.25     # TODO: what would be a good LPF cutoff?

    from mvpa2.misc.data_generators import autocorrelated_noise
    shambold = autocorrelated_noise(ds, sr, cutoff)

    # save to nifti file
    image = map2nifti(shambold)
    image.to_filename('{}.nii.gz'.format(outfile))
