---
nav_include: 5
title: Future Steps
---


## Future Steps

There are many things that we would like to do in future studies. The first one that comes to our mind is getting a better understanding of predictive power of each predictor. We could focus on particular categories (neuropsychological tests vs biomarkers vs imaging) or on particular tests within those categories (CDR vs FAQ vs MOCA). The goal would be to provide with some recommendations on the most cost effective tests or group of tests to the doctors and community. The way we could approach this is by getting a temporary slice were we have a substantial number of patient observations with tests in all categories and flattening it. We could use the median for some of the repeated tests if any in that time range. That will allow us to then play with the data by adding/removing some predictors and  fitting some of the models we have seen perform better, for example random forests.


Once the predictors are covered, another dimension we would have liked to explore is the time. We would do this by researching and applying some time series analysis approaches since this data is inherently of that nature.


For last we would like to improve the preprocessing of the data and to better tune our models. For a dataset with so many missing values a wise selection of observations and predictors, and thorough imputaton and preprocessing of the data is key for the accuracy of our models. We could try several imputation techniques like using random values or even predicting values by applying some simple models. Then when it comes to our models we feel we have a lot to tune and experiment with, specially we feel the neural networks have much more potential for this dataset that what we have seen.
