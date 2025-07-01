# Motivation
Fraudulent Transactions incur heavy, and sometimes unrecoverable, financial damage to credit card users. Detection of such transactions can help prevent financial loss of the victims and reduce financial crimes. Through the power of machine learning, it is possible to make use of the large amount of data available in order to detect such transactions.

# Data
Here, I made use of publicly available data for the detection of fraudulent transactions. The dataset used for this project is a .csv file and is publicly available on Kaggle at this link: 
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

This dataset has 284,807 transactions, consists of 30 features, and has a binary label: whether a transaction is fraudulent or not. All but two of the features in this dataset are the result of PCA (Principal Component Analysis). The two features are “time” and “amount” of the transactions. All the features that result from PCA are anonymized in order to prevent any possibility of tracing the financial data back to a client, thus, no information is available on what they present or mean. Regardless of the situation, I proceed to carry out the classification with the available information.

The transaction data is mostly clean and does not have major inconsistencies. Only few records have missing data, and since we have a large number of records, we can just discard those transactions and make use of the rest of the data.

It is worth noting that the data is highly skewed, since most of the transactions that occur are legitimate and only a tiny portion of them are fraudulent. This results in our dataset having 99.83% of its records in class 0 (legtimate transaction) and 0.17% of it in class 1 (fraudulent transaction).

# Method of Approach
The subtasks for this project are carried out in the order given below:
1. Prepare and process the data
1. Analyze the data
1. Model and Compare
1. Train and Test

Extra information about each step is given in the notebook for this project.