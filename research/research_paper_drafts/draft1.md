
Forecasting Valley Fever Incidence via Enhanced xLSTM Models Trained on Comprehensive Meteorological Data

# Abstract 

#####insert this somewhere in the abstract
We propose a simple question: How accurate can training predictive models become to correctly identify the weather pattersn that correlate to valley fever incidence?

# Keywords

# Introduction

Coccidioidomycosis, commonly known as Valley Fever (CM), is a fungal infection predominantly found in the southwestern United States, as well as parts of Central and South America. The infection is spread through spores released from soil disturbances, highlighting the direct influence of weather conditions on the dissemination of Coccidioides spores. Recent data indicates a consistent and significant increase in Valley Fever incidence each year in California. This increase impacts human health and imposes a considerable economic burden, with the total estimated cost reaching upwards of $700 million annually, and with no signs of incidence rates decreasing, treatment costs are expected to rise.

Our research endeavors to model and accurately predict the incidence rates of CM in California and apply this predictive accuracy to other similar geographic areas. We have utilized various temporal weather parameters, such as wind speed, to train multiple machine learning models to forecast annual CM incidence rates. Other studies have utilized regression models to establish a strong correlation between climatic variables and CM rates, particularly noting peak incidences during the fall season.

We propose the use of a modern machine learning architecture, the extended Long Short-Term Memory network (xLSTM), which incorporates residual memory blocks—a concept borrowed from Convolutional Neural Networks (CNNs). This approach allows the model to include more layers than previously feasible, addressing the notorious vanishing gradient problem that older LSTM models often faced. The xLSTM architecture enhances the traditional LSTM design, which itself improves upon the Recurrent Neural Network (RNN) framework by incorporating feedback signals for retrospection.

For our study, we implemented four different predictive models:

    A baseline regressor that predicts based on the mean of all outputs, ignoring the input.
    An LSTM model as a baseline metric for comparison of the xLSTM.
    An xLSTM model to assess if this new architecture can outperform traditional LSTMs in this domain.

In fields such as natural language processing, xLSTMs have already demonstrated substantial improvements over other state-of-the-art models, including transformers. LSTMs have been proven to excel in scenarios where limited data needs to be retained over extended periods [3]. A suitable feature for our dataset, which comprises 1,056 results divided among training, validation, and test sets.

# Methodologies

## Hardware Specifications
All studies were conducted using the same machine with a AMD 5950x 16 core 32 thread CPU with 128gb of DDR4 ram and a Nvidia 4090 24gb vram GPU. All sequential processes like data cleaning were ran on the CPU and all model training was ran on the GPU. 

## Data Sourcing
The data sourced to curate our datasets originates from 2 locations. For all of the data on the number or cases each county experienced was sourced from the California Department of Public Health (CDPH). This dataset included data from 64 locations spanning from 2001 to 2022. It features the counties name year nummber of cases and the rate. For our weather data we selected the 48 counties from the CDPH dataset that contained the most relevant number of cases per year. We then sourced a comprehensive dataset of hourly updated weather features spanning the entirety of the timeline of the other dataset. To do this we used the open weather api historical data call (https://openweathermap.org/api).

## Data Preperation
Several key descisions were made when it came to deciding on how to prepare the data. For the CDPH dataset we decided to not include the rate as an outcome variable due to it being flagged by the CDPH for being potentially unreliable. We also narrowed down our selection from the 64 locations to 48 due to certain locations not meeting a 60% threshold for containing data. With the weather dataset the original timespan was from 1979 to 2024 we filterred it out to only keep the years from 2001 to 2022 to match it with the CM cases dataset. We decided to not filter out any part of the year because, other studies have shown cases occur through the year and spike in September to November, thus including those weather patterns is pertinent. With the weather dataset we also chose to label encode 2 features that were werather descriptors, such as cloudy or sunny. With the weather dataset having updates on a hourly basis it would cause it to haev a sequense length of 8760, therefore we decided to aggregate the dataset down to an avearage instance per day redusing each instance to have a sequnce length of 365 Our descision to do this was due to  limited training data our mode would overfit the data quickly if it had to accommodate sequences any longer than that. Before shaping the final dataset we applied MinMax normalization to all of the weather data to decrease the large variances between different features. Resahping the daatacet to get it formated so that it would have the suitabale time sequences for the lstm models we arrived with a input shape of (1056, 365, 19) where the 1056 is the number of seqences, 365 is the size of each sequence, and 19 as the number of features per sequence. The output shape is (1056) for the 1056 possible outcomes. Then we made the descision to add data augmentation, but we did not want data contamination spreading fronm training to validation and tesing data. Therefore we split the dataset using random splitting to get the training lenght of 844, validation set of 158, and test set of 54. From there we increased the size of the trainig set by 20 times through trial and error of increasing the training set from 2 x to 100x we found 20x to be optimal for performance. For augmentation we created 20 tensors filled with random amounts and multiplied them by a noise variance from 0.01 to 0.00001. We user a logarithmic scale to create the varience of each noise level. See noise levels below. 

Noise Levels: [1.00000000e-02 6.95192796e-03 4.83293024e-03 3.35981829e-03
 2.33572147e-03 1.62377674e-03 1.12883789e-03 7.84759970e-04
 5.45559478e-04 3.79269019e-04 2.63665090e-04 1.83298071e-04
 1.27427499e-04 8.85866790e-05 6.15848211e-05 4.28133240e-05
 2.97635144e-05 2.06913808e-05 1.43844989e-05 1.00000000e-05]

Doing this worked very well yielding us with a final training set of length 17724, validation set of 158, and test set of 54.

## Models

### Baseline Regressor

It is important that in order to establish that our more advanced models are performant in learning the data we decided to establish a Baseline Regressor. This helps us have a start point to analyze  the lstm and xlstm models performance. A baseline regressor is a type of machine learning baseline model that is designed to ignore the input labels and only focusing on calctulating the mean of the target labels. With this we used the same dataset as we used for all of our models training it on the training set of len 4224 and testing its accuracy on the test set of 264 length. As a comparison of the quality for inference on all models we used the Root Mean Squarred Error (RMSE). Traingin of the baseline regressor took approximately 0.00045 seconds and resulted in a RSMS of 538.24. Since the baseline regressor only computes the mean of the targets there is no finetuning required for optimization of the model.

### LSTM

Broad Search results
Min Train RMSE: 28.356937058534083
Min Validation RMSE: 57.3488710034565
{'hidden_size': 256, 'num_layers': 2, 'bias': False, 'batch_first': True, 'dropout': 0, 'bidirectional': False, 'proj_size': 0}

keep this one in mind
200 2 True True 0.1 False 

Fine Search results
Min Train RMSE: 11.33083176851998
Min Validation RMSE: 70.89486087741568
{'hidden_size': 200, 'num_layers': 2, 'bias': False, 'batch_first': True, 'dropout': 0, 'bidirectional': False, 'proj_size': 0}


### xLSTM

# Results 

# Conclusion

# Adknowledgements

# References

[1]    al Sadeque, Z., & Bui, F. M. (2020). A Deep Learning Approach to Predict Weather Data Using Cascaded LSTM Network. Canadian Conference on Electrical and Computer Engineering, 2020-August. https://doi.org/10.1109/CCECE47787.2020.9255716

[2]  Beck, M., Pöppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M., Klambauer, G., Brandstetter, J., & Hochreiter, S. (2024). xLSTM: Extended Long Short-Term Memory. http://arxiv.org/abs/2405.04517

[3]     Gers, F. A., Schraudolph, N. N., & Schmidhuber, J. (2003). Learning precise timing with LSTM recurrent networks. Journal of Machine Learning Research, 3(1), 115–143. https://doi.org/10.1162/153244303768966139

[4]    Staudemeyer, R. C., & Morris, E. R. (2019). Understanding LSTM -- a tutorial into Long Short-Term Memory Recurrent Neural Networks. http://arxiv.org/abs/1909.09586

[5]  Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. http://arxiv.org/abs/1706.03762

[6]  Weaver, E. A., & Kolivras, K. N. (2018). Investigating the Relationship Between Climate and Valley Fever (Coccidioidomycosis). EcoHealth, 15(4), 840–852. https://doi.org/10.1007/s10393-018-1375-9

[7]  Wilson, L., Ting, J., Lin, H., Shah, R., Maclean, M., Peterson, M. W., Stockamp, N., Libke, R., & Brown, P. (2019). The rise of valley fever: Prevalence and cost burden of coccidioidomycosis infection in California. International Journal of Environmental Research and Public Health, 16(7). https://doi.org/10.3390/ijerph16071113

### Notes from reading papers that might be usefull to add as context of the paper

