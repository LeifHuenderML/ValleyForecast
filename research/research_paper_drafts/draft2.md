Coccidioidomycosis (CM), commonly known as Valley Fever, is a fungal infection caused by Coccidioides species that poses a significant public health challenge, particularly in the semi-arid regions of the Americas, with notable prevalence in California and Arizona. Previous epidemiological studies have established a correlation between CM incidence and regional weather patterns, indicating that climatic factors influence the fungus's life cycle and subsequent disease transmission. This study hypothesizes that Long Short-Term Memory (LSTM) and extended Long Short-Term Memory (xLSTM) models, known for their ability to capture long-term dependencies in time-series data, can outperform traditional statistical methods in predicting CM outbreak cases.
Our research analyzed daily meteorological features from 2001 to 2022 across 48 counties in California, encompassing diverse microclimates and CM incidence rates. The study evaluated 846 LSTM models and is currently assessing xLSTM models with various fine-tuning metrics. To ensure the reliability of our results, these advanced neural network architectures are cross-analyzed with Baseline Regression models and Multi-Layer Perceptrons, providing a comprehensive comparative framework.
Preliminary results indicate that LSTM-type architectures outperform traditional methods, demonstrating an increase in predictive accuracy of up to 87%. This improvement in predictive capability suggests a strong correlation between temporal microclimatic variations and regional CM incidence rates. The increased predictive power of these models has significant public health implications, potentially informing strategies for CM outbreak prevention and control. These findings contribute to the ongoing efforts to address CM, offering a new approach to understanding and potentially mitigating the impact of the disease in affected regions.

## Introduction

Coccidioidomycosis (CM), or Valley Fever, is a fungal disease that presents a growing challenge to public health systems in the Americas, particularly in arid and semi-arid regions. The causative agents, Coccidioides immitis and C. posadasii, are soil-dwelling fungi that release spores into the air when soil is disturbed, leading to respiratory infections in humans and animals. The disease's impact extends beyond individual health outcomes, affecting communities, healthcare systems, and local economies in endemic areas.
The spectrum of CM manifestations is broad and often unpredictable. While many infected individuals remain asymptomatic, others experience a range of symptoms from mild flu-like illness to severe pneumonia. In rare but serious cases, the infection can disseminate beyond the lungs, leading to meningitis, osteomyelitis, or cutaneous lesions. These severe forms of CM can result in long-term disability or even death, particularly among immunocompromised individuals, pregnant women, and certain ethnic groups who are at higher risk for disseminated disease.
Current strategies to combat CM are multifaceted but limited in their effectiveness. Early diagnosis relies on a combination of clinical suspicion, radiological findings, and serological tests, which can be challenging in non-endemic areas where awareness is low. Treatment typically involves antifungal medications such as fluconazole or itraconazole for symptomatic cases, with more aggressive therapies reserved for severe or disseminated disease. However, these treatments are often prolonged, costly, and associated with significant side effects. Prevention efforts primarily focus on dust control measures and public education campaigns to reduce exposure to fungal spores, but these approaches have shown limited success in curbing the rising incidence rates.
The economic impact of CM is substantial and growing. In California alone, the annual cost burden associated with CM is estimated to exceed $700 million, encompassing direct medical expenses, lost productivity, and ongoing care for chronic cases. With incidence rates showing a consistent upward trend, particularly in California and Arizona, there is an urgent need for more effective predictive models and preventive strategies.
Our research addresses this need by developing advanced machine learning models to forecast CM incidence rates based on comprehensive meteorological data. We focus on California, analyzing daily weather features from 2001 to 2022 across 48 counties, which represent a diverse range of microclimates and CM incidence patterns. By leveraging the power of Long Short-Term Memory (LSTM) networks and their extended variant (xLSTM), we aim to capture the complex, long-term dependencies between environmental factors and disease occurrence.
The choice of LSTM and xLSTM architectures is motivated by their proven efficacy in handling time-series data and their ability to retain information over extended periods. These characteristics are particularly relevant to our study, given the seasonal nature of CM and the potential lag between environmental conditions and disease manifestation. Our approach builds upon previous studies that have established correlations between climatic variables and CM rates using regression models, particularly noting peak incidences during the fall season following rainfall events.
By developing more accurate predictive models, our research aims to provide public health officials with a powerful tool for anticipating CM outbreaks and allocating resources more effectively. The potential applications of this work extend beyond California, offering a framework that could be adapted to other endemic regions facing similar challenges with fungal diseases influenced by environmental factors.
In the following sections, we detail our methodology, including data collection and preprocessing, model architecture, and evaluation metrics. We then present our results, discussing the performance of LSTM and xLSTM models in comparison to traditional statistical approaches and exploring the implications of our findings for CM prevention and control strategies.

## Methodologies
Our study employed a rigorous methodological approach encompassing data collection, preprocessing, model development, and evaluation. This section details the procedures and resources utilized to ensure reproducibility and validity of our findings.

## Hardware Specifications
To maintain consistency and enable efficient processing of large-scale meteorological and epidemiological data, all computational tasks were performed on a single, high-performance workstation. The system specifications were as follows:

Processor: AMD Ryzen 9 5950X (16 cores, 32 threads)
Memory: 128 GB DDR4 RAM
Graphics Processing Unit: NVIDIA GeForce RTX 4090 with 24 GB VRAM

This hardware configuration allowed for optimal distribution of computational tasks. CPU-intensive operations, such as data cleaning and preprocessing, were executed on the multi-core processor to leverage its parallel processing capabilities. The high-performance GPU was utilized exclusively for model training, taking advantage of its specialized architecture for deep learning computations. This division of labor ensured efficient utilization of resources and minimized processing time, particularly for the training of complex neural network architectures such as LSTM and xLSTM models.
The substantial RAM capacity (128 GB) facilitated the handling of large datasets in memory, reducing I/O operations and enhancing overall data processing speed. The high VRAM capacity of the GPU (24 GB) allowed for larger batch sizes during model training, potentially improving the stability and efficiency of the learning process.
By maintaining a consistent hardware environment across all experiments, we mitigated potential variations in performance that could arise from differing computational resources, thereby enhancing the reproducibility of our results.

## Data Sourcing
Our study utilized two primary data sources to construct a comprehensive dataset for analysis:
Epidemiological Data
The epidemiological data on Coccidioidomycosis (CM) cases were obtained from the California Department of Public Health (CDPH). This dataset encompassed:

Temporal range: 2001 to 2022
Geographical scope: 64 distinct locations within California
Key variables: County name, year, number of CM cases, and incidence rate

#### Meteorological Data
For the meteorological component, we focused on the 48 counties from the CDPH dataset that exhibited the most significant CM case numbers annually. This selection criterion ensured a robust representation of areas with varying CM prevalence.
The weather data were sourced using the OpenWeather API's historical data service (https://openweathermap.org/api). This comprehensive dataset included:

Temporal resolution: Hourly updates
Temporal range: Matched to the epidemiological data (2001-2022)
Geographical scope: 48 selected California counties

The OpenWeather API provided a rich set of meteorological variables, including but not limited to temperature, humidity, precipitation, wind speed, and atmospheric pressure. The high temporal resolution of this data allowed for a detailed analysis of weather patterns in relation to CM incidence.

#### Data Integration
The epidemiological and meteorological datasets were integrated based on temporal and geographical concordance. This integration allowed for a nuanced analysis of the relationship between weather patterns and CM incidence across diverse microclimates within California.
The use of these two complementary data sources enabled a multifaceted approach to understanding the environmental factors influencing CM incidence. By combining official public health records with high-resolution weather data, we aimed to capture both the epidemiological trends and the underlying environmental conditions that may influence the spread of Coccidioides species.
This data sourcing strategy provided a robust foundation for our subsequent analyses, allowing for the development of predictive models that incorporate both disease incidence patterns and detailed meteorological information.


## Data Preparation

The preparation of our dataset involved several crucial steps to ensure data quality, relevance, and suitability for our machine learning models. We implemented the following procedures:

### Epidemiological Data Refinement

1. **Outcome Variable Selection**: We excluded the incidence rate as an outcome variable due to reliability concerns flagged by the California Department of Public Health (CDPH).

2. **Geographical Filtering**: The initial 64 locations were reduced to 48 based on a data completeness threshold. Locations with less than 60% data availability were excluded to maintain dataset integrity.

### Meteorological Data Processing

1. **Temporal Alignment**: The weather dataset was filtered to match the epidemiological data timeframe (2001-2022), ensuring temporal consistency across all variables.

2. **Temporal Granularity Adjustment**: Hourly weather data was aggregated to daily averages, reducing the sequence length from 8,760 (hourly for a year) to 365 (daily for a year). This adjustment mitigated potential overfitting issues due to limited training data while maintaining a sufficiently detailed temporal resolution.

3. **Feature Encoding**: Categorical weather descriptors (e.g., cloudy, sunny) were label-encoded to facilitate numerical analysis.

4. **Normalization**: MinMax normalization was applied to all weather features to standardize the scale of different meteorological variables, enhancing model performance and stability.

### Dataset Structuring

The final dataset was shaped to accommodate the requirements of LSTM models:
- Input shape: (1056, 365, 19)
  - 1056: Number of sequences (year-location combinations)
  - 365: Days per year
  - 19: Number of features per day
- Output shape: (1056) corresponding to the CM case counts for each sequence

### Data Augmentation

To enhance model robustness and mitigate overfitting, we implemented a data augmentation strategy:

1. **Dataset Splitting**: The dataset was randomly split into training (844 sequences), validation (158 sequences), and test (54 sequences) sets.

2. **Augmentation Procedure**: The training set was augmented 20-fold, a multiplier determined through empirical optimization.

3. **Noise Injection**: Augmentation was achieved by adding Gaussian noise to the original data. The noise variance was scaled logarithmically from 1e-2 to 1e-5, creating 20 distinct noise levels:

   ```python
   noise_levels = np.logspace(-2, -5, 20)
   ```

4. **Final Dataset Composition**:
   - Training set: 17,724 sequences (844 * 21, including original data)
   - Validation set: 158 sequences
   - Test set: 54 sequences

This augmentation strategy significantly expanded our training data while preserving the integrity of our validation and test sets, crucial for unbiased model evaluation.

The resulting dataset provided a robust foundation for training our LSTM and xLSTM models, balancing the need for substantial training data with the imperative to maintain distinct validation and test sets for reliable model assessment.

Let's refine and expand the Models section to provide a more comprehensive and academic description of your approach:

## Models

Our study employed a hierarchical approach to model development, implementing four distinct predictive models. This strategy allowed for a systematic comparison of performance across different levels of model complexity and architectural designs. The models implemented were:

1. Baseline Regressor
2. Multilayer Perceptron (MLP)
3. Long Short-Term Memory (LSTM) Network
4. Extended Long Short-Term Memory (xLSTM) Network

### Baseline Regressor

We implemented a simple baseline regressor as a fundamental benchmark. This model predicts the mean of all output values, disregarding input features. While rudimentary, this approach provides a critical reference point for assessing the performance gains achieved by more sophisticated models.

### Multilayer Perceptron (MLP)

The MLP serves as an intermediate model, bridging the gap between the baseline regressor and more complex recurrent architectures. This feedforward neural network provides insight into the capability of traditional neural networks to capture the underlying patterns in our dataset. The MLP's performance offers a valuable comparison point, helping to quantify the benefits of employing recurrent architectures for this time-series prediction task.

### Long Short-Term Memory (LSTM) Network

The LSTM model represents our primary recurrent neural network architecture. Known for its ability to capture long-term dependencies in sequential data, the LSTM is well-suited for our task of predicting CM incidence based on time-series meteorological data. This model serves as a strong baseline for assessing the performance of more advanced recurrent architectures.

### Extended Long Short-Term Memory (xLSTM) Network

The xLSTM is a novel extension of the traditional LSTM architecture, incorporating innovations such as exponential gating, memory mixing, and matrix memory. We hypothesize that these enhancements may allow the xLSTM to more effectively capture the complex, long-term relationships between meteorological patterns and CM incidence. By comparing the xLSTM's performance to that of the traditional LSTM, we aim to evaluate the potential benefits of these architectural innovations in the context of our specific prediction task.

### Model Comparison Strategy

Our approach to model comparison is designed to provide a comprehensive understanding of each architecture's strengths and limitations:

1. The baseline regressor establishes a minimum performance threshold.
2. The MLP demonstrates the capability of non-recurrent neural networks in capturing relevant patterns.
3. The LSTM serves as a strong baseline for recurrent architectures, leveraging its ability to model long-term dependencies.
4. The xLSTM allows us to assess whether the latest innovations in recurrent neural network design offer tangible benefits for our specific prediction task.

By implementing this diverse set of models, we aim to not only identify the most effective approach for predicting CM incidence but also to gain insights into the nature of the relationship between meteorological factors and disease occurrence. This comprehensive comparison strategy enables us to make informed recommendations about the most suitable modeling approaches for similar epidemiological prediction tasks.

Let's refine this section to make it more academic and detailed:

### Baseline Regressor

To establish a fundamental performance benchmark, we implemented a Baseline Regressor. This model serves as a crucial reference point for evaluating the efficacy of our more sophisticated models in capturing the underlying patterns in the data.

#### Methodology

The Baseline Regressor is designed to ignore input features and predict a constant value for all instances. This value is computed as the mean of the target variable in the training set. Formally, for a set of n training examples with target values {y₁, y₂, ..., yₙ}, the Baseline Regressor predicts:

ŷ = (1/n) ∑ᵢ yᵢ

This approach provides a naive prediction that serves as a lower bound for model performance.

#### Implementation Details

- **Training Data**: The model was trained on the same dataset used for all other models, consisting of 4,224 instances.
- **Test Data**: Model performance was evaluated on a test set of 264 instances.
- **Evaluation Metric**: Root Mean Squared Error (RMSE) was employed as the primary metric for assessing prediction quality across all models. RMSE is defined as:

  RMSE = √[(1/n) ∑ᵢ (yᵢ - ŷᵢ)²]

  where yᵢ are the true values and ŷᵢ are the predicted values.

#### Results

- **Training Time**: Approximately 0.00045 seconds
- **RMSE**: 538.24

The exceptionally short training time is attributed to the model's simplicity, requiring only the computation of the mean target value.

#### Significance

The Baseline Regressor's performance provides a critical context for interpreting the results of more complex models. Any model that fails to outperform this baseline significantly would be considered inadequate for the task at hand. Conversely, the degree to which other models improve upon this baseline RMSE of 538.24 serves as a quantitative measure of their effectiveness in capturing the underlying patterns in the data.

It's worth noting that due to its design, the Baseline Regressor does not require hyperparameter tuning or optimization. This characteristic underscores its role as a fixed reference point against which the performance gains of more sophisticated models can be measured.

Let's refine and expand this section to provide a more comprehensive and academic description of the MLP model:

### Multilayer Perceptron (MLP)

#### Theoretical Background

A Multilayer Perceptron (MLP) is a class of feedforward artificial neural network characterized by at least three layers of nodes: an input layer, one or more hidden layers, and an output layer. The MLP's architecture facilitates the learning of non-linear relationships in data through the use of activation functions and backpropagation for weight adjustment.

The primary strengths of MLPs lie in their ability to:
1. Approximate complex, non-linear functions
2. Handle high-dimensional input spaces
3. Learn hierarchical representations of data

#### Model Architecture

For our study, we implemented a 3-layer MLP with the following structure:

1. Input Layer: Dimensionality matching our feature space
2. Hidden Layer: With Rectified Linear Unit (ReLU) activation
3. Output Layer: Single node for regression

The ReLU activation function, defined as f(x) = max(0, x), was chosen for its computational efficiency and effectiveness in mitigating the vanishing gradient problem.

#### Training Process

The model was trained using the following parameters:

- **Epochs**: 1500
- **Optimizer**: Adaptive Moment Estimation (Adam)
- **Learning Rate**: 5e-5
- **Loss Function**: Mean Squared Error (MSE)

The Adam optimizer was selected for its ability to adapt the learning rate during training, potentially leading to faster convergence and improved performance on complex loss landscapes.

#### Performance Evaluation

The MLP's performance was evaluated using the Root Mean Squared Error (RMSE) metric, consistent with our baseline regressor evaluation. The final results yielded an RMSE of 115.1793.

#### Results Analysis

The MLP's performance (RMSE: 115.1793) represents a substantial improvement over the baseline regressor (RMSE: 538.24), indicating that:

1. There are significant non-linear relationships in the data that the MLP is able to capture.
2. The chosen architecture and hyperparameters are well-suited to the problem domain.

The magnitude of improvement (approximately 78.6% reduction in RMSE) suggests that the meteorological features contain strong predictive power for CM incidence, which the MLP is able to leverage effectively.

#### Significance

The MLP's performance serves as a crucial intermediate benchmark between our baseline regressor and more complex recurrent models. It demonstrates the potential of neural networks to capture relevant patterns in our dataset, while also providing a reference point for assessing the additional value that recurrent architectures might bring to the task.

The strong performance of the MLP also suggests that some of the relevant patterns for predicting CM incidence may not necessarily require long-term temporal dependencies, as the MLP processes each time step independently. This insight will be valuable when interpreting the performance of our recurrent models in subsequent analyses.

Let's refine and expand this section to provide a more comprehensive and academic description of the LSTM model:

### Long Short-Term Memory (LSTM) Network

#### Theoretical Background

Long Short-Term Memory (LSTM) networks are a specialized form of Recurrent Neural Networks (RNNs) designed to address the vanishing gradient problem inherent in traditional RNNs. LSTMs are particularly well-suited for processing and predicting time series data due to their ability to capture long-term dependencies.

The key innovation of LSTMs is the introduction of a memory cell, which allows the network to selectively remember or forget information over extended sequences. This mechanism is implemented through three primary gates:

1. Input gate: Controls the flow of new information into the cell state
2. Forget gate: Determines what information should be discarded from the cell state
3. Output gate: Regulates the information output from the cell state

These gates enable LSTMs to maintain and update relevant information over long sequences, making them particularly effective for tasks involving time-series data.

#### Model Architecture Optimization

To determine the most effective LSTM configuration for our CM incidence prediction task, we conducted an extensive hyperparameter search:

- **Search Space**: 846 distinct model configurations
- **Search Method**: Cardinal search (a form of grid search)

The optimal architecture identified through this process consists of:

1. Two LSTM layers with a hidden size of 256 units each
2. A two-layer Multilayer Perceptron (MLP) with ReLU activation functions
3. 10% dropout rate applied between all layers except the final output layer

This architecture combines the sequential learning capabilities of LSTMs with the non-linear function approximation of MLPs, potentially capturing both temporal dependencies and complex feature interactions.

#### Training Process

The model was trained using:
- **Optimizer**: Adam (learning rate and other hyperparameters determined during the search process)
- **Loss Function**: Mean Squared Error (MSE)
- **Epochs**: [Number of epochs, if available]

#### Performance Evaluation

The optimized LSTM model achieved an RMSE of 57.34887, representing a significant improvement over both the baseline regressor (RMSE: 538.24) and the MLP model (RMSE: 115.1793).

#### Results Analysis

The LSTM's performance demonstrates:

1. A 89.3% reduction in RMSE compared to the baseline regressor
2. A 50.2% reduction in RMSE compared to the MLP model

These improvements suggest that:

a) Temporal dependencies play a crucial role in predicting CM incidence
b) The LSTM's ability to capture long-term patterns in the meteorological data significantly enhances predictive accuracy

#### Significance

To our knowledge, this study represents the first application of LSTM networks for predicting CM incidence using temporal meteorological data. The substantial performance improvement over non-recurrent models underscores the importance of considering long-term weather patterns in CM prediction.

The success of this LSTM architecture provides a strong foundation for further exploration of recurrent models in epidemiological forecasting, particularly for diseases with environmental risk factors.



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


