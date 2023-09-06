# ecg-data-wrangler
The data used for training and testing is in the "output" folder.
The directory balanced contains the balanced training files used in the final soltuion
The directory unbalanced contains the unbalanced training files used for experimentation and test files used for final solution
The directories patients_N and patients_L contain the data for the N and L classifiers for each individual patient used to produce figure 14 and 16 respectively

To run training and test, install BioHEL in Linux and run
biohel conf.conf <train-file.conf> <test-file.conf> 
To produce data from scratch run src\ecg\main.py

