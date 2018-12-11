# 24787B_Final_Project

The current files are associated with the final project for class 24-787B - Fall 18, at CMU. 

The preprocessing functions are located on functions.py. The neural network functionality has been implemented on neural.py. 
Additional Jupyter Notebooks have been also made available to show intermediate results achieved throughout the course of the project.

Needless to say, preprocessing took a considerable amount of time and space. The original database was downloaded from the CAP Sleep 
Database (https://physionet.org/pn6/capslpdb/) and had to be converted to intermediate files because, for some reason, the original files
would not always open. The initial intermediate files were pickled versions of the edf files.

Further pre-processing also required extracting the labels from text files, grabbing the time stamps, and synchronizing the labels to
the datasets. While there are many reasons to resample the data to 1 Hz, one of them is because the labels are sampled at 1 Hz. In any case
we created functions to upsample the labels at higher frequencies based on a nearest neighbor algorithm combined with a function to expand
the labels. Label expansion is necessary because the native format indicates the sleep stage, and the amount of time in seconds that the
patient will be on this stage. What the function does is expand those seconds and make sure that the stage is correct for that period of
time. In the case of overlaps, we favored A-phase stages over Sleep stages.

Finally we combined the processed features and labels into a single csv file per person per night. This finally allowed for quicker data
processing and some manageable level of iterations on the side of the neural network design. Additional functions were later added to
save and restore trained models for future analysis and reporting. These models are not included on this folder.

