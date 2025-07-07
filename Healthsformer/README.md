# Motivation
Synthetic data is data that is artificially generated rather than gathered by recording real-world events.
This type of data is used in a variety of fields as a substitute for data that if released, would
compromise the privacy of the subjects or the confidentiality of the data. For instance, in the field of
healthcare or finance there exist many datasets that can have valuable or personally identifiable information
(PIIs) within them, and cannot be released to the public. Synthetic data can evade the privacy issues that arise
from using real subject (consumer, patient, etc.) information without permission or compensation, and provide
researchers with data that closely represent the original data. 

The goal of this project is to implement one of the recent synthetic sequential data generation methods
designed for financial data to generate synthetic sequential healthcare data, for patients suffering from
ischemic stroke. This synthetic data can be disclosed to researchers in order to allow for optimization of
patient treatments to save as many lives as possible.

# Dataset
The dataset used in this project is the version 2.2 of the Medical Information Mart for Intensive Care,
abbreviated as MIMIC-IV. This dataset is a de-identified dataset that provides critical care data for
more than 40,000 patients admitted to intensive care units at BIDMC.

# Approach
Of the many methods and tools available for synthetic data generation, I make use of a family of
them called transformers. Transformers are neural networks that aim to undertake sequential tasks
while handling long-range dependencies with ease through a mechanism called attention. This mechanism is the
backbone of the large language models (LLMs) that are popular today. This implementation is based on the 
architecture proposed in the paper "Banksformer: A Deep Generative Model for Synthetic Transaction Sequences".
For the purpose of synthetic data generation, an autoregressive decoder-only transformer architecture is
implemented, which consists of three broad layer types: input layer, decoder layers, and the output layers.

# Findings
- The decoder-only transformer architecture is capable of capturing the sequential relationships within the data with high efficiency and accuracy.
- Transformers are particularly data-hungry and complex models that require significant amount of data to train on.
- Without a sufficiently large dataset, regardless of the number of parameters of the models, transformers tend to overfit and perform poorly.
