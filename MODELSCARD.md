# Model Card
## Project Context

We are currently at the third stage of the project. In the preceding steps, we conducted web scraping from a reputable website specializing in property sales and rentals. Subsequently, we analyzed the scraped data. Presently, our focus lies on constructing a model aimed at assisting realtors in predicting property prices in Belgium.

## Data

The primary dataset utilized for training purposes is accessible in the data folder under the name properties.csv. The target variable for prediction is the property price. As of now, no subsets of the dataset have been created for different models. The final features included in the model are as follows:

    property_type: Categorized as Apartment or House.
    subproperty_type: Further subdivisions within the property types.
    region: Divided into Walloon or Flanders regions.
    total_area_sqm: Total living area in square meters.
    surface_land_sqm: Total land area in square meters.
    nbr_bedrooms: Number of bedrooms.
    fl_furnished: Boolean indicating if the property is furnished.
    terrace_sqm: Size of terrace in square meters (0 if not present).
    garden_sqm: Size of garden in square meters (0 if not present).
    fl_swimming_pool: Boolean indicating the presence of a swimming pool.
    fl_floodzone: Boolean indicating if the building is located in a flood zone.
    state_building: State of the building.
    primary_energy_consumption_sqm: Energy consumption.
    postal_zone: First two digits of the postcode.

However, further testing is required to refine the model.

## Model Details

The models currently under evaluation are all available within the scikit-learn module. As of now, the models tested include linear regression and random forest regression. The random forest regression model has been chosen as the final model due to consistently higher performance compared to linear regression.

## Performance

The performance metrics utilized for evaluation are R2 (coefficient of determination) and MSE (mean squared error). Additionally, predictions have been made against both the test and train sets, and the actual values have been mapped against the predicted values for comparison.

Linear Regression:
![Linear Regression](images\linear_regression.png)

Neural Network:
![Neural Network Regression](images\neural_network.png)

Random Forest Regression:
![Random Forest Regression](images\random_forest_regression.png)

## Limitations

The model is built upon a subset where prices are under €1.2M due to insufficient data points for higher prices, which would have otherwise skewed the model.

Some limitations of the model include:

    Overfitting: Visual examination of the Random Forest Regression model reveals a good fit for the training set, which does not generalize well to the test set, indicating overfitting.

## Usage

This model has been implemented using Python, with the following modules:

    pandas: for handling dataframes
    scikit-learn: for creating pipelines and training models
    pickle: for saving the pipeline

The training script is available in the train.py file. The workflow involves:

    Data preprocessing, which includes removing certain columns and rows with missing values.
    Splitting the data into training and test sets.
    Building a pipeline to one-hot encode categorical data, standardize numerical values, and impute missing values using KNN.
    Training the model using Random Forest Regression.

To train the model, simply execute the train.py file:
~~~
python train.py
~~~

This will generate the required pickle files for making predictions.

To generate predictions, provide a CSV file formatted similarly to properties.csv (without the price column) in the data folder named predict.csv. Then run the predict.py file:
~~~
python predict.py
~~~
The output will be a prediction.csv file in the main folder, containing the original data with predictions appended.

## Maintainers

Jens Dedeyne: [LinkedIn](https://www.linkedin.com/in/jens-dedeyne/)