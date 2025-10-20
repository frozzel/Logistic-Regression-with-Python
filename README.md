# Logistic Regression with Python

## Objectives

* Use Logistic Regression for classification
* Preprocess data for modeling
* Implement Logistic regression on real world data

## Install and import the required libraries

Make sure the required libraries are available by executing the cell below.
```bash
pip install numpy==2.2.0
pip install pandas==2.2.3
pip install scikit-learn==1.6.0
pip install matplotlib==3.9.3
```

### Screenshot

Insert a screenshot illustrating results or workflow:

![Model output screenshot](images/screenshot.png)

## Classification with Logistic Regression

### Scenario
Assume that you are working for a telecommunications company which is concerned about the number of customers leaving their land-line business for cable competitors. They need to understand who is more likely to leave the company.

###  Load the Telco Churn data 
Telco Churn is a hypothetical data file that concerns a telecommunications company's efforts to reduce turnover in its customer base. Each case corresponds to a separate customer and it records various demographic and service usage information. Before you can work with the data, you must use the URL to get the ChurnData.csv.

### About the dataset
We will use a telecommunications dataset for predicting customer churn. This is a historical customer dataset where each row represents one customer. The data is relatively easy to understand, and you may uncover insights you can use immediately. Typically it is less expensive to keep customers than acquire new ones, so the focus of this analysis is to predict the customers who will stay with the company. 
<br><br>
This data set provides you information about customer preferences, services opted, personal details, etc. which helps you predict customer churn.

