# Project description
This project is aimed at the analysis of the sentiments of users who purchased beauty products from Amazon during 2023. The dataset used for this task can be found on the huggingface website under this name: McAuley-Lab/Amazon-Reviews-2023 or at this link: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

# Objectives for sentiment analysis:
- Understand Customer Opinions: Identify and analyze the emotions, opinions, and attitudes expressed in customer reviews.
- Classify Sentiment: Automatically categorize text as positive or negative.
- Monitor Brand Reputation: Track how people feel about a brand, product, or service over time to detect trends or issues early.
- Improve Decision-Making: Provide actionable insights to marketing, product development, and customer service teams.
- Automate Feedback Analysis: Reduce manual effort by using models to quickly process large volumes of textual data.

# Plan of action
- The preprocessing of texts will be done by two different methods
    1. TF-IDF
    2. Text embedding
- The sentiment analysis task will be classified by different classification models.
- A final best model will be selected based on computational expenses, accuracy, inference time, ...

# Summary and conclusions
- Model comparisons are done through 5-fold cross validation scores.
- The best model was found to be Text Embedding (with the all-MiniLM-L6-v2 model) + Linear SVM classifier with a cross validation score of 86.43%.
- The next best model was found to be TF-IDF + Logistic Regression with a cross validation score of 85.94%.
- The TF-IDF method seems to offer lower computational costs due to not needing GPUs, while offering competitive performance.
- The most suitable model was selected as TF-IDF + Logistic Regression and offered a test accuracy of 87.98%.
- The same methods can be used for different families of products to observe and track user opinions on products and brands over time.