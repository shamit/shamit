## shamit

ATM - a small python project to simulate (fmri) data.

Momentarily, it is still aimed at simulating 4D fMRI data containing expected signal.

Basic functionality is in shamit/shamit.py. The spiritual basis are the scripts from an fMRI data analysis projects,
which can be found in /famface_original_scripts


## Target use cases for future development

In the long run it would be nice to fake/simulate not only fMRI (BOLD) data but
also of other modalities.  High level targets for simulated data will be

- H0 data - no signal
- hypothetical effect - simulated response to the experimental design with clearly
  defined and thus known properties (design, SNR, etc). So some portions of data
  will still be carrying H0 (no signal) and others carry desired response
  
  
### H0 data generation

The goal is to have H0 data indistinguishable from a "real" data -- produced 
dataset should look like a "real" (original) one but carry no signal of interest.
Various approaches should (ideally) be implemented

- random - data in the core is produced by RNG but should carry the 
  characteristics of original data, e.g. 
  - reusing the base image for fMRI 4D data
  - have the same "motion"
  - have the same temporal and spatial (smoothness) characteristics  
- injected - taking data from another dataset (ideally with similar acquisition 
  parameters) and injecting it pretending to be the data for the current 
  study
  - just replace original subjects with subjects from another
    dataset while retaining some specific original data intact (_tasks, or just 
    participants.tsv)
  - replace data selectively (e.g. func/*_bold.{json,nii.gz}*) with data
    from other studies, e.g. via "proper" spatial realignment
- permutted - permutting across subjects/sessions

###  Hypothetical effect

For BOLD data could be as simple as producing target response using linear model
specification, which e.g. could originally be specified for doing GLM analysis 
(e.g. as in BIDS linear models/PyBIDS/fitlins specification) 

In the long run integration with VirtualBrain to produce realistic signal
(possibly for EEG/MEG as well) given more thorough specification would be great.  