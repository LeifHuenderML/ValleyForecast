
Forecasting Valley Fever Incidence via Enhanced xLSTM Models Trained on Comprehensive Meteorological Data

# Abstract 

# Keywords

# Introduction
CM aka Valley Fever is a fungal infection located in the southwest united states, and portions of central and south america(The Rise of Valley Fever Prevalence and Cost Burden of Coccidioidomycosis Infection in California). The root cause is through spores being released from soil disturbances making the weather have a direct effect on the spread of Coccidioides spores ((papers/Investigating the Relationship Between Climate and Valley Fever (Coccidioidomycosis).pdf)). Each year in california the rates have been on a steady and steep rise. (include here more info about how the disease affects humans) From a study conducted in 2017 they found that the total estimated cost burden of CM is upwards of 700 million dollars, with no signs of slowing incidences the total treatment costs for this will surely continue to rise. Our research is an attempt to model and accuartely predict the rates of CM in california and extend that prediction accuracy to other closely tied geographic locations. We used temporal weather parameters such as wind speed to train several machine learning models to predict the rates of CM that will occur in that year. Early studies from (papers/Investigating the Relationship Between Climate and Valley Fever (Coccidioidomycosis).pdf) used regression models to find strong links between climatic variables and the rates of CM. Incidences were reported to be the highest in the fall (Investigating the Relationship Between Climate and Valley Fever (Coccidioidomycosis). We propose that a modern machine learning (ML) architecture called extended long short-term memory networks(xLSTM) will prove valuable in accurately forecasting rates of CM incidences due to its incorporation of residual memory blocks a concept derived from Convolutional Neural Networkst (CNN) where it will allow the model to become deeper (meaning more layers) than was previously thought. This incorporation of residual blocks aims to solve the issue of the infamous vanishing gradient problem where the lstm model would struggle with being deeper because the restcted flow of gradients from one layer to the next. The xLSTM architecure is a extension of the LSTM architecture which istelf is a continuation of the Reccurrent Neural Network (RNN) architectur that had feedback signasl alloing it to look back in time. RNNs where flawed due to their common issue of vanishing gradients ([text](<../../papers/Understanding LSTM – a tutorial into Long Short-Term Memory Recurrent Neural Networks.pdf>)). For our study we conducted four difrent predictive models a dummy regressor which does not consider the input and only predicts the mean of all the outputs, a linear regressor that tries to fis a predictive line to the input outpu pairs, a lstm and a xlstm to see if the current new architectur is capable of outperforming lstm in this field. In natural language processing the xlstm has allready been seend to have a substantiol imporvement over other state of the art model architecture such as the transformer ([text](<../../papers/Understanding LSTM – a tutorial into Long Short-Term Memory Recurrent Neural Networks.pdf>)) (/home/intellect/Documents/Research/Current/ValleyForecast/papers/Attention Is All You Need.pdf). 

# Methodologies

## Hardware Specifications
All studies were conducted using the same machine with a AMD 5950x 16 core 32 thread CPU with 128gb of DDR4 ram and a Nvidia 4090 24gb vram GPU. All sequential processes like data cleaning were ran on the CPU and all model training was ran on the GPU. 

# Results 

# Conclusion

# Adknowledgements

# References

### Notes from reading papers that might be usefull to add as context of the paper

