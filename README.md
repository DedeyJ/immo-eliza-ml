# immo-eliza-ml
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
# Property Price Prediction Model

## Overview

This project aims to develop a model for predicting property prices in Belgium. The model is designed to assist realtors in estimating property prices based on various features such as property type, region, area size, number of bedrooms, and more.

## Project Context

We are currently at the third stage of the project. In the preceding steps, we conducted web scraping from a reputable website specializing in property sales and rentals. Subsequently, we analyzed the scraped data. Presently, our focus lies on constructing a model aimed at assisting realtors in predicting property prices in Belgium.

## Repo Structure

├─data
│   └─ properties.csv
├─images
├─model
├─predict.py
├─train.py
├─MODELSCARD.md
└─README.md

## Performance

The performance metrics utilized for evaluation are R2 (coefficient of determination) and MSE (mean squared error). Additionally, predictions have been made against both the test and train sets, and the actual values have been mapped against the predicted values for comparison.

**Linear Regression:**
- ![Linear Regression](images\linear_regression.png "RMSE and visuals")

**Random Forest Regression:**
- ![Random Forest](images\random_forest_regression.png "RMSE and visuals")


## Usage

install requirements using:
~~~
pip install requirements.txt
~~~

To train the model, simply execute the `train.py` file:
~~~
python train.py
~~~

This will generate the required pickle files for making predictions, which can be found in the model folder.

To generate predictions, provide a CSV file formatted similarly to `properties.csv` (without the price column) in the `data` folder named `predict.csv`. Then run the `predict.py` file:

~~~
python predict.py
~~~


The output will be a `prediction.csv` file in the main folder, containing the original data with predictions appended.

## Maintainers

Jens Dedeyne:
- [LinkedIn](https://www.linkedin.com/in/jens-dedeyne/)


## Timeline

This project has been done within a span of five days.