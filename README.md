# Predict blood donation
This project works with data collected from the donor database of Blood Transfusion Service Center in Hsin-Chu City in Taiwan. The center passes its blood transfusion service bus to one university in Hsin-Chu City to gather blood donated about every three months. The dataset, obtained from the UCI Machine Learning Repository, consists of a random sample of 748 donors. The task is to predict if a blood donor will donate within a given time window. The work contains a full model-building process: from inspecting the dataset to using the tpot library to automate your Machine Learning pipeline.

## Project rundown

* Inspecting transfusion.data file
* Loading the blood donations data
* Inspecting transfusion DataFrame
* Creating target column
* Checking target incidence
* Splitting transfusion into train and test datasets
* Selecting model using TPOT
* Checking the variance
* Log normalization
* Training the linear regression model

## Tech stack
* Shell
* Python 3
* Pandas
* scikit-learn
* TPOT library

_The project is based on a datacamp assignment published by Dimitri Denisjonok._
