
Forecasting Valley Fever Incidence via Enhanced xLSTM Models Trained on Comprehensive Meteorological Data

# Abstract 

#####insert this somewhere in the abstract
We propose a simple question: How accurate can training predictive models become to correctly identify the weather pattersn that correlate to valley fever incidence?

# Keywords

# Introduction

Coccidioidomycosis, commonly known as Valley Fever (CM), is a fungal infection predominantly found in the southwestern United States, as well as parts of Central and South America. The infection is spread through spores released from soil disturbances, highlighting the direct influence of weather conditions on the dissemination of Coccidioides spores. Recent data indicates a consistent and significant increase in Valley Fever incidence each year in California. This increase not only impacts human health but also imposes a considerable economic burden, with the total estimated cost reaching upwards of $700 million annually, and with no signs of incidence rates decreasing, treatment costs are expected to rise.

Our research endeavors to model and accurately predict the incidence rates of CM in California and apply this predictive accuracy to other similar geographic areas. We have utilized various temporal weather parameters, such as wind speed, to train multiple machine learning models to forecast annual CM incidence rates. Other studies have utilized regression models to establish a strong correlation between climatic variables and CM rates, particularly noting peak incidences during the fall season.

We propose the use of a modern machine learning architecture, the extended Long Short-Term Memory network (xLSTM), which incorporates residual memory blocks—a concept borrowed from Convolutional Neural Networks (CNNs). This approach allows the model to include more layers than previously feasible, addressing the notorious vanishing gradient problem that older LSTM models often faced. The xLSTM architecture enhances the traditional LSTM design, which itself improves upon the Recurrent Neural Network (RNN) framework by incorporating feedback signals for retrospection.

For our study, we implemented four different predictive models:

    A dummy regressor that predicts based on the mean of all outputs, ignoring the input.
    A linear regressor that fits a predictive line to the input-output pairs.
    An LSTM model.
    An xLSTM model to assess if this new architecture can outperform traditional LSTMs in this domain.

In fields such as natural language processing, xLSTMs have already demonstrated substantial improvements over other state-of-the-art models, including transformers. LSTMs have been proven to excel in scenarios where limited data needs to be retained over extended periods [3]. A suitable feature for our dataset, which comprises 1,056 results divided among training, validation, and test sets.

# Methodologies

## Hardware Specifications
All studies were conducted using the same machine with a AMD 5950x 16 core 32 thread CPU with 128gb of DDR4 ram and a Nvidia 4090 24gb vram GPU. All sequential processes like data cleaning were ran on the CPU and all model training was ran on the GPU. 

## Data Sourcing
The data sourced to curate our datasets originates from 2 locations. For all of the data on the number or cases each county experienced was sourced from the California Department of Public Health (CDPH). This dataset included data from 64 locations spanning from 2001 to 2022. It features the counties name year nummber of cases and the rate. For our weather data we selected the 48 counties from the CDPH dataset that contained the most relevant number of cases per year. We then sourced a comprehensive dataset of hourly updated weather features spanning the entirety of the timeline of the other dataset. To do this we used the open weather api historical data call (https://openweathermap.org/api).

## Data Preperation
Several key descisions were made when it came to deciding on how to prepare the data. For the CDPH dataset we decided to not include the rate as an outcome variable due to it being flagged by the CDPH for being potentially unreliable. We also narrowed down our selection from the 64 locations to 48 due to certain locations not meeting a 60% threshold for containing data. With the weather dataset the original timespan was from 1979 to 2024 we filterred it out to only keep the years from 2000 to 2022, our descision for going from 2000 is because for the 2001 rates from the CDPH we wanted to train the models to learn the weather paterns from the previous years rates leading up to the current years prediciton. We decided to filter the year spanning from September 1 to August 31 encomassing 365 day of weather data for the model to learn. Our descision for this cutoff period was to allow the model to be able to forecast early predictions before fall when the rates are the highest for CM. With the weather dataset we also chose to one hot encode 2 features that were werather descriptors, such as cloudy or sunny. Before shaping the final dataset we applied MinMax normalization to all of the weather data to decrease the large variances between different features. Resahping the daatacet to get it formated so that it would have the suitabale time sequences for the lstm models we arrived with a input shape of (1056, 8760, 69) where the 1056 is the number of seqences, 8760 is the size of each sequence 365 days * 24 hours per day, and 69 as the number of features per sequence. The output shape is (1056) for the 1056 possible outcomes. Finally we created an additional dataset from this dataset but with augmented data added causing the data set to grow by 4x. Our method for augmenting the data was to apply a noise of 0.01 and 0.001 to the inpud data and pairing it up with the same outcomes. This resulted in an input of (4224, 8760, 69) and a output of (4224). 

## Data Splitting

## Models

### Baseline Regressor

Dataset 1 RMSE: 477.99225 Time: 0.00042
Dataset 2 RMSE: 299.98972 Time: 0.00053

### LSTM

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

