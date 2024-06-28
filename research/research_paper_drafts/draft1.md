Forecasting Coccidioidomycosis incidence via enhanced LSTM models trained on comprehensive meteorological data

# Abstract
Coccidioidomycosis (CM), commonly known as Valley Fever, is a fungal infection caused by Coccidioides species. It is predominantly found in semi-arid regions of the Americas, particularly in California and Arizona, and poses a significant public health challenge. Previous research has established a strong correlation between CM incidence and regional weather patterns, suggesting that climatic factors significantly affect the fungus's life cycle and subsequent disease transmission. We hypothesize that Long-Short-Term Memory (LSTM) and extended Long-Short-Term Memory (xLSTM) models, known for their ability to capture long-term dependencies in time-series data, can surpass traditional statistical methods in predicting outbreak cases. Our research analyzed daily weather features from 2001 to 2022 across 48 counties in California, encompassing a wide range of microclimates and CM incidence rates. The study evaluated 846 LSTM models and currently is assessing xLSTM models with various fine-tuning metrics. These advanced models are cross-analyzed with Baseline Regression models and Multi-Layer Perceptrons to provide a comprehensive comparison, ensuring the reliability of our results. Preliminary results indicate that LSTM-type architectures significantly outperform traditional methods, demonstrating an increase in accuracy of up to 87%. This marked improvement in predictive capability strongly suggests a robust correlation between temporal microclimates and regional CM incidence rates. The increased predictive capability of these models has significant public health implications. These findings inform strategies for addressing CM outbreaks, potentially leading to more effective prevention and control measures in the ongoing efforts to combat the disease.

# Keywords

# Introduction
Coccidioidomycosis, commonly known as Valley Fever (CM), is a fungal infection predominantly found in the southwestern United States, as well as parts of Central and South America. The infection is spread through spores released from soil disturbances, highlighting the direct influence of weather conditions on the dissemination of Coccidioides spores. Recent data indicates a consistent and significant increase in Valley Fever incidence each year in California. This increase impacts human health and imposes a considerable economic burden, with the total estimated cost reaching upwards of $700 million annually, and with no signs of incidence rates decreasing, treatment costs are expected to rise.

Our research endeavors to model and accurately predict the incidence rates of CM in California and apply this predictive accuracy to other similar geographic areas. We have utilized various temporal weather parameters, such as wind speed, to train multiple machine learnig models to forecast annual CM incidence rates. Other studies have utilized regression models to establish a strong correlation between climatic variables and CM rates, particularly noting peak incidences during the fall season after rainfall.

We propose the use of a Long Short-Term Memory (LSTM) network which has gained popularity when dealing with temporal machine learning tasks due to its intrinsic design to learn long term sequential dependencies. LSTMs have proven their usefullness in other machine learning tasks wher they are tasked with learning weather patterns to predict regressable outcomes. We also venture into unexplored models with the extended long short term memory (xLSTM) a new model that introduces exponential gating, memory mixing and matrix memory. The incorporation of exponential gating we think that it should be able to outperform the original LSTM architecture when it comes to retaining long sequential informatin which is essential for our use case given that our dataset sequence length is 365 instances long spaning each day of the year. In fields such as natural language processing, xLSTMs have already demonstrated substantial improvements over other state-of-the-art models, including transformers. LSTMs have been proven to excel in scenarios where limited data needs to be retained over extended periods [3]. A suitable feature for our dataset, which comprises 1,056 results divided among training, validation, and test sets.

# Methodologies

## Hardware Specifications
All studies were conducted using the same computer with a AMD 5950x 16 core 32 thread CPU with 128gb of DDR4 ram and a Nvidia 4090 24gb vram GPU. All sequential processes like data cleaning were ran on the CPU and all model training was ran on the GPU. 

## Data Sourcing
The data sourced to curate our datasets originates from 2 locations. For all of the data on the number or cases each county experienced was sourced from the California Department of Public Health (CDPH). This dataset included data from 64 locations spanning from 2001 to 2022. It features the counties name year nummber of cases and the rate. For our weather data we selected the 48 counties from the CDPH dataset that contained the most relevant number of cases per year. We then sourced a comprehensive dataset of hourly updated weather features spanning the entirety of the timeline of the other dataset. To do this we used the open weather api historical data call (https://openweathermap.org/api).

## Data Preperation
Several key descisions were made when it came to deciding on how to prepare the data. For the CDPH dataset we decided to not include the rate as an outcome variable due to it being flagged by the CDPH for being potentially unreliable. We also narrowed down our selection from the 64 locations to 48 due to certain locations not meeting a 60% threshold for containing data. With the weather dataset the original timespan was from 1979 to 2024 we filterred it out to only keep the years from 2001 to 2022 to match it with the CM cases dataset. We decided to not filter out any part of the year because, other studies have shown cases occur through the year and spike in September to November, thus including those weather patterns is pertinent. With the weather dataset we also chose to label encode 2 features that were werather descriptors, such as cloudy or sunny. With the weather dataset having updates on a hourly basis it would cause it to haev a sequense length of 8760, therefore we decided to aggregate the dataset down to an avearage instance per day redusing each instance to have a sequnce length of 365 Our descision to do this was due to  limited training data our mode would overfit the data quickly if it had to accommodate sequences any longer than that. Before shaping the final dataset we applied MinMax normalization to all of the weather data to decrease the large variances between different features. Resahping the daatacet to get it formated so that it would have the suitabale time sequences for the lstm models we arrived with a input shape of (1056, 365, 19) where the 1056 is the number of seqences, 365 is the size of each sequence, and 19 as the number of features per sequence. The output shape is (1056) for the 1056 possible outcomes. Then we made the descision to add data augmentation, but we did not want data contamination spreading fronm training to validation and tesing data. Therefore we split the dataset using random splitting to get the training lenght of 844, validation set of 158, and test set of 54. From there we increased the size of the trainig set by 20 times through trial and error of increasing the training set from 2 x to 100x we found 20x to be optimal for performance. For augmentation we created 20 tensors filled with random amounts and multiplied them by a noise variance from 0.01 to 0.00001. We user a logarithmic scale to create the varience of each noise level. See noise levels below. 

###incorporate hte logarithmic algorithm instead####

Noise Levels: [1.00000000e-02 6.95192796e-03 4.83293024e-03 3.35981829e-03
 2.33572147e-03 1.62377674e-03 1.12883789e-03 7.84759970e-04
 5.45559478e-04 3.79269019e-04 2.63665090e-04 1.83298071e-04
 1.27427499e-04 8.85866790e-05 6.15848211e-05 4.28133240e-05
 2.97635144e-05 2.06913808e-05 1.43844989e-05 1.00000000e-05]

Doing this worked very well yielding us with a final training set of length 17724, validation set of 158, and test set of 54.

## Models

For our study, we implemented four different predictive models:
    A baseline regressor that predicts based on the mean of all outputs, ignoring the input.
    A Multilayer Perceptron (MLP) to act as a middle varriant between understanding how well the LSTM architectures perform compared to a traditional neural network.
    An LSTM model as a baseline metric for comparison of the xLSTM.
    An xLSTM model to assess if this new architecture can outperform traditional LSTMs in this domain.

### Baseline Regressor

It is important that in order to establish that our more advanced models are performant in learning the data we decided to establish a Baseline Regressor. This helps us have a start point to analyze  the other models performance. A baseline regressor is a type of machine learning baseline model that is designed to ignore the input labels and only focusing on calctulating the mean of the target labels. With this we used the same dataset as we used for all of our models training it on the training set of len 4224 and testing its accuracy on the test set of 264 length. As a comparison of the quality for inference on all models we used the Root Mean Squarred Error (RMSE). Traingin of the baseline regressor took approximately 0.00045 seconds and resulted in a RSMS of 538.24. Since the baseline regressor only computes the mean of the targets there is no finetuning required for optimization of the model.

### MLP
A MLP is a type of artificial neural network composed of at least 3 layers of neuraons (input, hidden, output). It is charaterized by its feed forward architecture where information flows unidirectionally from its input to its output. It is very useful for many big data tasks by excelling at learning lon linear relationships in data through the uset of activation functions and backpropagations. We opted to use a 3 layer neural network with the relu activation function. See figure below for the forward pass. The mlp was trained over 1500 epochs with an ADAM optimizer set to a learning rate of 5e-5. The final results yielded a RMSE of 115.1793 which performed better then we expected based on how well the baseline regressor performed.

### LSTM
The LSTM has long been a strong performing model when it comes to times series tasks due to its design. They are a type of recurrent neural netowrk architecture designed to address the vanighing gradient problem that traditional rnns faced. The lstm introduses the memory cell which allows the network to retain important data for extended sequences. Although their popularity they have not been previously sude to predict the incidence of CM from temporal meteorological data. To find the most effective model we conducted a cardinal searhc over the search space of 846 model hyperparameters to establish that the most effective model contains 2 lstm layers with a hidden size of 256 feeding into a 2 layer mlp with relu activtion functions and a 10% dropout rate between all layers but the last. This resulted in lower rmse of 57.34887. See the forward pass below for more details.

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


