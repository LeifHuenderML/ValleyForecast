# Summary of `prep.py`

## Overview
The `prep.py` script is responsible for loading, cleaning, merging, and transforming weather and health datasets into a format suitable for machine learning models, particularly for forecasting with an LSTM neural network.

## Operations Performed

### Data Cleaning and Preparation
1. **Rates Dataset:**
   - Loads an Excel file containing Valley Fever cases and incidence rates.
   - Renames columns for clarity.
   - Drops unreliable 'rates' column and unnecessary first and last rows.
   - Standardizes 'county' names to lowercase.

2. **Weather Dataset:**
   - Loads a CSV file containing weather data.
   - Drops unnecessary columns and fills missing values.
   - Converts date-time information for further processing.
   - Encodes categorical weather descriptions using label encoding.
   - Aggregates weather data to daily averages and prepares for merging by adjusting 'county' names and splitting date into year, month, and day components.

### Data Merging
- Merges the cleaned weather and rates datasets based on 'county' and 'year'.
- Filters data to include only entries up to August 31st of each year to forecast early cases of Valley Fever.

### Feature Scaling and Preparation for LSTM
- Applies Min-Max normalization to various weather-related features.
- Segments the data into sequences suitable for LSTM input, with data limited to the first 243 days per group.
- Enhances the dataset by adding noise-augmented versions of the data to increase its size and robustness.

### Dataset Splitting and Saving
- Splits the augmented dataset into training, validation, and test sets with an 80-15-5 percent split.
- Saves these datasets to disk for use in training and evaluating the LSTM model.

## Additional Notes
- The script includes multiple print statements for tracking progress and outcomes, such as data shapes and successful data saving steps.
- Utilizes PyTorch and Scikit-learn libraries for data handling and preprocessing.

