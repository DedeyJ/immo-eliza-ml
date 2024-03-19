import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle


file_name = r".\data\properties.csv"
df = pd.read_csv(file_name)

# Making a pipeline class to put in scikit pipeline
class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop, rows_to_drop, condition):
        self.columns_to_drop = columns_to_drop
        self.rows_to_drop = rows_to_drop
        self.condition = condition

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        # Replace "MISSING" values with None
        # for col in df.columns:
        #     df.loc[df[col] == "MISSING", col] = None

        # Extract postal zone from zip code
        df['postal_zone'] = df['zip_code'].astype(str).str[:2]

        # Drop specified columns
        df = df.drop(labels=self.columns_to_drop, axis=1)

        # Drop rows with missing values
        df = df.dropna(subset=self.rows_to_drop)

        # Conditional replacement of surface_land_sqm
        df.loc[df['property_type'] == self.condition, 'surface_land_sqm'] = df.loc[df['property_type'] == self.condition, 'total_area_sqm']

        return df



# Decide which columns to drop
columns_to_drop = ["nbr_frontages", "id", "zip_code", "locality","latitude","longitude","fl_terrace","fl_garden","cadastral_income","fl_double_glazing","construction_year","equipped_kitchen", "province", "epc", "heating_type", "fl_open_fire"]
rows_to_drop = ["terrace_sqm", "garden_sqm","primary_energy_consumption_sqm","total_area_sqm", "state_building"]
condition = "APARTMENT"

# Define pipeline
pipeline = Pipeline([
    ('preprocessor', Preprocessor(columns_to_drop, rows_to_drop, condition))
])

save_preprocess = pipeline.fit(df)

# Save the pipeline to possibly reuse
with open('preprocess.pkl', 'wb') as f:
    pickle.dump(save_preprocess, f)

df = save_preprocess.transform(df)
# Split data into features (X) and target (y)
X = df.drop(columns=["price"])
y = df["price"]

#Split X and y into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=50, test_size=0.2)

# Find the numerical columns
numerical_columns = X_train.select_dtypes(include=['int', 'float']).columns

# Find the categorical columns
categorical_columns = X_train.select_dtypes(include=['object']).columns

# Create pipelines for numerical and categorical transformations
numerical_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('imputer', KNNImputer(n_neighbors=5)) # You can adjust n_neighbors as needed
])

categorical_pipeline = Pipeline([
    ('onehot', OneHotEncoder())
])

# Combine numerical and categorical pipelines using ColumnTransformer
preprocessor = ColumnTransformer([
    ('numerical', numerical_pipeline, numerical_columns),
    ('categorical', categorical_pipeline, categorical_columns)
])

params =  {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20]}  # Number of trees and depth
model = RandomForestRegressor()
# Create the final pipeline
final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# Fit the pipeline (this will select the best model automatically)
training = final_pipeline.fit(X_train, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(training, f)

with open('model.pkl', 'rb') as f:
    input = pickle.load(f)

print(input.score(X_test, y_test))
