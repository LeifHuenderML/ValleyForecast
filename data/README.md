# Data Documentation 

This document serves as a centralized location for tracking updates and changes made to datasets and what each one goes to

# Edited Datasets

## valley_fever_rates

data/valley_fever_rates/valley_fever_rates_05_30_24.csv

    - this dataset uses the data obtained from [5/30/24 original dataset from https://www.cdph.ca.gov/Programs/CID/DCDC/Pages/ValleyFeverDashboard.aspx]
    (original_datasets/ValleyFeverDashboard_Data.xlsx) but it is made into a csv 

    this process happened on 05/30/24 the file that did the editing can be found [here](../src/data_cleaning/original_valley_fever_cleaning.ipynb) 

    the changes mate to it include:
        # the first row doesnt contain data so i am dropping it 
        #the bottom row doesnt contain data too so i am dropping it 
        #changed the labels into more consice labels
        #change the dtype of year and cases to ints
        #change the dtype of rate to floats



# Original Datasets

##  California Valley Fever Dataset

- [5/30/24 original dataset from https://www.cdph.ca.gov/Programs/CID/DCDC/Pages/ValleyFeverDashboard.aspx]
(original_datasets/ValleyFeverDashboard_Data.xlsx)

this is the original dataset obtained on 5/30/24 with no changes made to it








