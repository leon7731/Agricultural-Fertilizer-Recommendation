import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation


import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Step 1: Loading Dataset
# Dataset Source: Kaggle
# Link: https://www.kaggle.com/datasets/sanchitagholap/crop-and-fertilizer-dataset-for-westernmaharashtra 
df = pd.read_csv('Data/Crop and fertilizer dataset.csv')

# Removing leading and trailing spaces from column names
df.columns = df.columns.str.strip()

# Replace multiple spaces with a single space in all column names
df.columns = df.columns.str.replace(r'\s+', ' ', regex=True)  

# Drop Unnecessary Columns
df.drop(['District_Name', 'Link'], axis=1, inplace=True)
# Step 2: Overview of Dataset
df.info()
df.head()
# Step 3: EDA - Missing Values Analysis 
## Step 3)i): EDA - Show Missing Values in each Column
# Get percentage of null values in each column
null_values_percentage = df.isnull().mean().round(4).mul(100).sort_values(ascending=False)
print('-' * 44)
print("Percentage(%) of null values in each column")
print('-' * 44)
print(null_values_percentage)
print('\n')

# Get total null values in each column
total_null_values = df.isnull().sum().sort_values(ascending=False)
print('-' * 33)
print("Total null values in each column")
print('-' * 33)
print(total_null_values)
# Step 4: EDA - Duplicate Values Analysis 
## Step 4)i): EDA - Show Duplicate Values Rows
# Get percentage of duplicate rows
total_rows = len(df)
duplicate_rows = df.duplicated().sum()
duplicate_percentage = (duplicate_rows / total_rows) * 100

print('-' * 48)
print("Percentage(%) of duplicate rows in the DataFrame")
print('-' * 48)
print(f"{duplicate_percentage:.2f}%")
print('\n')

# Get total number of duplicate rows
print('-' * 30)
print("Total number of duplicate rows")
print('-' * 30)
print(duplicate_rows)

# Step 5: EDA - Analyzing Column
## Step 5)i): EDA - Univariate Analysis
def univariate_analysis_plotly(df):
    """

    Perform univariate analysis on a DataFrame using Plotly.

    Parameters:
    - df: DataFrame to be analyzed.

    Returns:
    - Interactive Plotly plots with summary statistics in the legend.
    """
    colors = px.colors.qualitative.Plotly

    for idx, column in enumerate(df.columns):

        # Generate descriptive statistics
        stats = df[column].describe()
        stats_str = '<br>'.join([f'{k}: {v:.2f}' if isinstance(v, (float, int)) else f'{k}: {v}' for k, v in stats.items()])

        # Visualization based on datatype
        if np.issubdtype(df[column].dtype, np.number):

            # If the column is numeric, plot a histogram with a box plot as marginal
            fig = px.histogram(df, x=column, marginal="box", title=f"Histogram for {column}", color_discrete_sequence=[colors[idx % len(colors)]])
        else:

            # If the column is categorical or textual, plot a bar chart
            value_counts = df[column].value_counts()

            fig = px.bar(x=value_counts.index, y=value_counts.values, 
                         title=f"Bar Chart for {column}", 
                         labels={"x": column, "y": "Count"},
                         color_discrete_sequence=[colors[idx % len(colors)]])

        # Add descriptive stats as a legend using a dummy trace for both types of columns
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", 
                                 name=stats_str, showlegend=True, 
                                 hoverinfo="none", opacity=0))
        
        fig.show()

univariate_analysis_plotly(df)
# Step 6): EDA - Feature Selection
## Step 6)i): EDA - Heatmap

def apply_auto_ordinal_encoding(df: pd.DataFrame, columns_to_encode: list[str]) -> pd.DataFrame:
    """
    Apply automatic Ordinal Encoding to specific columns of a DataFrame.

    Parameters:
    - df: Input DataFrame
    - columns_to_encode: List of column names to apply Ordinal Encoding

    Returns:
    - DataFrame with Ordinally Encoded columns
    """
    
    df_encoded = df.copy()
    
    for column in columns_to_encode:
        unique_values = df[column].unique()
        ordinal_mapping = {key: val for val, key in enumerate(unique_values)}
        
        # Print the ordinal mapping for the column
        print(f"Ordinal Encoding for '{column}': {ordinal_mapping}")
        
        df_encoded[column] = df[column].map(ordinal_mapping)
    
    return df_encoded

:
df_encoded_eda = apply_auto_ordinal_encoding(df, ['Soil_color', 'Crop', 'Fertilizer', ])
df_encoded_eda.head()
import pandas as pd
import plotly.figure_factory as ff


def heatmap_correlations(df: pd.DataFrame, targetVariable:str ,colorscale:str="Viridis"):
    """
    Create a heatmap showing the correlation of all pairs of variables in the dataframe.
    Parameters:
    - df (pd.DataFrame): Data to be plotted.
    - targetVariable (str): The dependent variable for which correlations will be displayed.
    - colorscale (str): Desired colorscale for the heatmap. Default is "Viridis".
    Returns:
    - None: Shows the heatmap.
    """
    
    # Filtering only numerical columns
    df_numeric = df.select_dtypes(include=['number'])
    
    # Check if targetVariable is numeric
    if targetVariable not in df_numeric.columns:
        print(f"The target variable {targetVariable} is not numeric.")
        return
    
    # Print the correlation of the target variable with other variables
    print('-' * 52)
    print(f"Correlation of {targetVariable} with other Independent variables")
    print('-' * 52)
    print(df_numeric.corr()[targetVariable].sort_values(ascending=False))
    
    # Calculate the correlation matrix
    corr_matrix = df_numeric.corr()
    # Create a heatmap using the correlation matrix
    fig = ff.create_annotated_heatmap(
        z=corr_matrix.values, 
        x=list(corr_matrix.columns), 
        y=list(corr_matrix.index),
        annotation_text=corr_matrix.round(2).values,
        colorscale=colorscale
    )

    fig.update_layout(title="Correlation Heatmap of Variables")
    
    fig.show()



)
heatmap_correlations(df_encoded_eda, targetVariable="Fertilizer", colorscale='RdYlGn')

"""
----------------------------------------------------
Correlation of Fertilizer with other Independent variables
----------------------------------------------------
Fertilizer     1.000000
Crop           0.459263
Soil_color     0.125628
Potassium     -0.044753
pH            -0.051754
Rainfall      -0.103483
Temperature   -0.124724
Phosphorus    -0.174454
Nitrogen      -0.214226
Name: Fertilizer, dtype: float64

"""


# # Drop All Columns with Less 10% Positive and Negative Correlation
# df.drop(["Potassium",
#          "pH"
#         ], axis=1, inplace=True)

# df.head()
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# compute the vif for all given features
def compute_vif(dataframe: pd.DataFrame, numerical_columns:list, sort_ascending:bool=True):
    """
    Calculate the Variance Inflation Factor (VIF) for each feature in a DataFrame.
    
    Parameters:
    - dataframe (pd.DataFrame): The DataFrame containing the features.
    - numerical_columns (list): The list of numerical columns to calculate VIF for.
    - sort_ascending (bool): Whether to sort the VIF scores in ascending order. Default is True.
    """
    
    X = dataframe[numerical_columns]
    # the calculation of variance inflation requires a constant
    X['intercept'] = 1
    
    # create dataframe to store vif values
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['Variable']!='intercept']
    
    # Sort the VIF data
    vif.sort_values(by="VIF", ascending=sort_ascending, inplace=True)
    
    return vif


# Get Numerical Features
numerical_features = df.select_dtypes(include=np.number).columns.tolist()  # Get all numeric columns


# compute vif 
compute_vif(
    dataframe=df,
    numerical_columns=numerical_features,
    sort_ascending=False)

# Step 7): EDA - Feature Engineering/Scaling 
 - Categorical Feature Engineering/Scaling
 - Numerical Feature Engineering/Scaling
df.head()
# def apply_auto_ordinal_encoding(df: pd.DataFrame, columns_to_encode: list[str]) -> pd.DataFrame:
#     """
#     Apply automatic Ordinal Encoding to specific columns of a DataFrame.

#     Parameters:
#     - df: Input DataFrame
#     - columns_to_encode: List of column names to apply Ordinal Encoding

#     Returns:
#     - DataFrame with Ordinally Encoded columns
#     """
    
#     df_encoded = df.copy()
    
#     for column in columns_to_encode:
#         unique_values = df[column].unique()
#         ordinal_mapping = {key: val for val, key in enumerate(unique_values)}
        
#         # Print the ordinal mapping for the column
#         print(f"Ordinal Encoding for '{column}': {ordinal_mapping}")
        
#         df_encoded[column] = df[column].map(ordinal_mapping)
    
#     return df_encoded


# df = apply_auto_ordinal_encoding(df, ["Soil_color", "Crop", "Fertilizer"])
# df.head()
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define preprocessing for numerical and categorical features
numerical_features = ["Nitrogen", "Phosphorus", "Potassium", "pH", "Rainfall", "Temperature"]
categorical_features = ["Soil_color", "Crop", "Fertilizer"]

# Preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_features),
        ('cat', OrdinalEncoder(), categorical_features)
       
      
    ])
# Step 8) Train Test Split
df['Fertilizer'].value_counts().sort_index()  
### Train Test Split - SMOTENC
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split


X = df.drop("Fertilizer", axis=1)
y = df["Fertilizer"]


categorical_features_indices = [5]

# Apply SMOTE-NC
smote_nc = SMOTENC(sampling_strategy='auto', 
                   random_state=42, 
                   k_neighbors=5, 
                   n_jobs=-1,
                   categorical_features=categorical_features_indices
                   )

X_resampled, y_resampled = smote_nc.fit_resample(X, y)

# Split the resampled data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Generate distinct colors for each class using Plotly Express color scales
def generate_colors(n):
    color_scale = px.colors.qualitative.Plotly
    return [color_scale[i % len(color_scale)] for i in range(n)]

# Plot the class distribution before and after resampling
fig = make_subplots(rows=1, cols=2, subplot_titles=("Original Class Distribution", "Resampled Class Distribution after SMOTE"))

# Original class distribution
original_counts = y.value_counts().sort_index()
fig.add_trace(go.Bar(x=original_counts.index, y=original_counts.values, marker_color=generate_colors(len(original_counts)), showlegend=False), row=1, col=1)

# Resampled class distribution
resampled_counts = y_resampled.value_counts().sort_index()
fig.add_trace(go.Bar(x=resampled_counts.index, y=resampled_counts.values, marker_color=generate_colors(len(resampled_counts)), showlegend=False), row=1, col=2)

# Update layout
fig.update_layout(title_text="Class Distribution Before and After SMOTE", xaxis_title="Exited", yaxis_title="Count")

# Show plot
fig.show()
# Step 9) MLP Classifier Pipeline Model
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def Confusion_Matrix_For_Multi_Class_With_Overview(title, y_test, y_pred):
    """
    Create a confusion matrix for multi-class classification with detailed overview.
    Parameters:
    - title: Title for the confusion matrix plot.
    - y_test: True labels of the test data.
    - y_pred: Predicted labels of the test data.
    Returns:
    - A seaborn heatmap representing the confusion matrix.
    """
    # Creating the confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)

    # Determine class labels
    class_labels = np.unique(np.concatenate((y_test, y_pred)))

    # Calculate the counts and percentages for the confusion matrix
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    
    # Calculate TP and FP percentages
    TP_percentages = ["{0:.2%}".format(value/np.sum(cf_matrix, axis=1)[i]) for i, value in enumerate(np.diag(cf_matrix))]
    FP_percentages = ["{0:.2%}".format((np.sum(cf_matrix, axis=0)[i] - value)/np.sum(cf_matrix)) for i, value in enumerate(np.diag(cf_matrix))]
    
    # Combine TP and FP with their percentages
    combined_info = []
    for i in range(cf_matrix.shape[0]):
        for j in range(cf_matrix.shape[1]):
            value = cf_matrix[i, j]
            if i == j:  # True Positive
                combined_info.append(f"{value}\n(TP: {TP_percentages[i]})")
            else:  # False Positive
                combined_info.append(f"{value}\n(FP: {FP_percentages[j]})")
    labels = np.asarray(combined_info).reshape(cf_matrix.shape)

    # Plotting the heatmap
    plt.figure(figsize=(25, 25))
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    ax.set_title(f'{title}\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values')

    # Show the plot
    plt.show()

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib

# Best Hyperparameters
"""
Best Parameters: {
    'n_layers': 2, 
    'hidden_layer_size_1': 28, 
    'activation': 'tanh', 
    'solver': 'adam', 
    'alpha': 3.867181696498066e-05, 
    'learning_rate': 'constant', 
    'learning_rate_init': 0.01600185284869121, 
    'max_iter': 496
    }
    
Best Score: 0.9142873590327213
"""

# Best parameters found from Optuna
best_params = {
    'n_layers': 2, 
    'hidden_layer_size_1': 28, 
    'activation': 'tanh', 
    'solver': 'adam', 
    'alpha': 3.867181696498066e-05, 
    'learning_rate': 'constant', 
    'learning_rate_init': 0.01600185284869121, 
    'max_iter': 496
}


hidden_layer_sizes = tuple([best_params['hidden_layer_size_1']] * best_params['n_layers'])


# Create a pipeline that first transforms the data then fits the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MLPClassifier(
                        hidden_layer_sizes=hidden_layer_sizes,
                        activation='tanh',
                        solver='adam',
                        alpha=3.867181696498066e-05,
                        batch_size='auto', 
                        learning_rate = 'constant',
                        learning_rate_init=0.016,
                        max_iter=496,
                        random_state=42))
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Save the model to a file
joblib_file = "Model/mlp_model_pipeline.joblib"
joblib.dump(pipeline, joblib_file)


# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model using the custom confusion matrix function
Confusion_Matrix_For_Multi_Class_With_Overview("Confusion Matrix", y_test, y_pred)

# Creating Classification Report
print(classification_report(y_test, y_pred))

# 10) MLP Classifier Saved Model Prediction
import joblib
import pandas as pd

# Load the saved model
joblib_file = "Model/mlp_model_pipeline.joblib"
loaded_model = joblib.load(joblib_file)

# Load the ordinal encoding mappings
ordinal_mappings_path = "Mappings/ordinal_mappings.joblib"
ordinal_mappings = joblib.load(ordinal_mappings_path)

# Create reverse mappings for labels
reverse_mappings = {v: k for k, v in ordinal_mappings['Fertilizer'].items()}

# New data for prediction
new_data = pd.DataFrame({
    'Soil_color': ["Black"],
    'Nitrogen': [75],
    'Phosphorus': [50],
    'Potassium': [100],
    'pH': [6.5],
    'Rainfall': [1000],
    'Temperature': [20],
    'Crop': ["Sugarcane"]
})

# Transform the new data and make predictions
new_predictions = loaded_model.predict(new_data)

# Convert the predictions back to the original labels
predicted_labels = [reverse_mappings[pred] for pred in new_predictions]

print("Predictions for new data (encoded):", new_predictions)
print("Predictions for new data (original labels):", predicted_labels)
