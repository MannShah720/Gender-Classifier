# Gender Classifier

A simple ML project that predicts whether a person is male or female based on height, weight and shoe size (using mock data).

## What i've learnt
- Using a train/test split to train models on 80% of the data and evaluate it on the rest, allowing the model to learn patterns and perform better on unseen data.
- Using `random_state` to make train/test splits reproducible.
- Using Pandas to convert the mock data from a regular list into a DataFrame, allowing me to organise the features into labelled columns.
- Comparing the prediction and accuracy of the following models: Decision Tree, Logisitic Regression & K-NN
- Experimenting with K-NN's `n_neighbors` hyperparameter and seeing how changing it can affect predictions.

However, this project has limitations due to the very small & simplified dataset. The data may not generalize to real-world populations, meaning the predictions and accuracy scores should not be considered reliable.
