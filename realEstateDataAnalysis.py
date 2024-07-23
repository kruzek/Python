# Author Samuel Krúžek, samuel.kruzek@gmail.com

# Comments from the author:
""" 
In this file, Jyputer code cells have been used, creating small independently executable parts of code.

For running the code , please assign your local path of 'datasetx'
to the variable sales_data_path.
"""
#%%
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
#%%
# Import dataset
sales_data_path = r"C:\Users\skruzek\Downloads\salesData.csv"
dataset_sales = pd.read_csv(sales_data_path)
# Clean dataset 
dataset_sales = dataset_sales.drop_duplicates()
dataset_sales = dataset_sales.drop(columns=['garden_area', 'terrace_area', 'balcony_area', 'external_id'])
pd.DataFrame(dataset_sales).dropna()
# %%
# Get rid of the rows, where price value is not present, or the price is in different format
dataset_sales['is_valid'] = dataset_sales['price'].str.match(r'^\d+(\.\d+)?$')
dataset_sales=dataset_sales[dataset_sales['is_valid']==True]
# Clean the exterior column from the curly brackets 
dataset_sales['exterior_area']=dataset_sales['exterior_area'].str.replace(r'}', '')
dataset_sales['exterior_area']=dataset_sales['exterior_area'].str.replace(r'{', '')
# %%
# Convert price and exterior_area column to float 
dataset_sales['price'] = dataset_sales['price'].astype(float)
dataset_sales['exterior_area'] = dataset_sales['exterior_area'].astype(float)

# Drop the temporary column is_valid
dataset_sales.drop(columns=['is_valid'], inplace=True)
# %%
# Feature Engineering (creating new variable)
dataset_sales['price_p_sqr_meter'] = dataset_sales['price'] / dataset_sales['floor_area']

# %%
# Function for textual analyses of topN in chosen column
# groupByVariable is the column name to analyze, topN is number of top values shown, nameOfVariable is  nickname of the column you want to analyze in string
# addComparingStats shows you for every shown Top value comparing to the second and third best in their category
def Display_sales_performance_for_topN_textually(groupByVariable, topN, dataset, nameOfVariable, addComparingStats):
    # Calculate the total price of units sold by each groupByVariable
    total_price = dataset.groupby(groupByVariable)['price'].sum()

    # Determine the top N groupByVariables that sold the most units overall
    top_groupByVariables = total_price.nlargest(topN).index.tolist()

    # Loop through the top N groupByVariables
    for i, top_groupByVariable in enumerate(top_groupByVariables, 1):
        # Calculate the total price of units sold by the top groupByVariable
        total_price_top_groupByVariable = total_price[top_groupByVariable]

        # Calculate the total price of units sold by the second and third best groupByVariable
        # Exclude the top groupByVariable from the calculation
        sorted_groupByVariable_total_price = total_price.drop(top_groupByVariable).sort_values(ascending=False)
        second_best_groupByVariable = sorted_groupByVariable_total_price.index[0]
        third_best_groupByVariable = sorted_groupByVariable_total_price.index[1]
        total_price_second_best_groupByVariable = total_price[second_best_groupByVariable]
        total_price_third_best_groupByVariable = total_price[third_best_groupByVariable]

        # Calculate the performance of the top groupByVariable compared to the second and third best groupByVariables in percentage
        performance_second_best_groupByVariable = (total_price_top_groupByVariable - total_price_second_best_groupByVariable) / total_price_second_best_groupByVariable * 100
        performance_third_best_groupByVariable = (total_price_top_groupByVariable - total_price_third_best_groupByVariable) / total_price_third_best_groupByVariable * 100

        
        if addComparingStats == True:
            # Print the summary of sales performance for each top groupByVariable
            print(f"Summary of Sales Performance for the {i}th best {nameOfVariable} '{top_groupByVariable}':")
            print(f"Total price of units sold: {total_price_top_groupByVariable}")
            print(f"Performance compared to second best {nameOfVariable} ({second_best_groupByVariable}): {performance_second_best_groupByVariable:.2f}%")
            print(f"Performance compared to third best {nameOfVariable} ({third_best_groupByVariable}): {performance_third_best_groupByVariable:.2f}%")
            print()

    summary_data = []
    for top_groupByVariable in top_groupByVariables:
        # Calculate the total price of units sold by the top groupByVariable
        total_price_top_groupByVariable = total_price[top_groupByVariable]
        
        # Calculate the number of units sold by the top groupByVariable
        num_units_sold = dataset[dataset[groupByVariable] == top_groupByVariable]['price'].count()
        
        summary_data.append({
            f'{nameOfVariable}': top_groupByVariable,
            'Revenue': total_price_top_groupByVariable,
            'Number of Units Sold': num_units_sold
        })

    summary_table = pd.DataFrame(summary_data)

    # Print the summary table
    print(f"Summary of Sales Performance for the top {topN} {nameOfVariable}s:")
    print(summary_table)

# Function for graphical analyses of topN in chosen column
# groupByVariable is the column name to analyze, topN is number of top values shown
def Display_sales_performance_for_topN_inBarChart(groupByVariable, topN, dataset):
    # Convert 'date_sold' column to datetime format
    dataset['date_sold'] = pd.to_datetime(dataset['date_sold'])

    # Extract year and month from 'date_sold' column
    dataset['year'] = dataset['date_sold'].dt.year
    dataset['month'] = dataset['date_sold'].dt.month

    # Calculate the total price of units sold by each groupByVariable
    total_price = dataset.groupby(groupByVariable)['price'].sum()

    # Determine the top N groupByVariables that sold the most units overall
    top_groupByVariables = total_price.nlargest(topN)

    # Create a bar plot of revenue for the top N projects
    plt.figure(figsize=(10, 6))
    top_groupByVariables.plot(kind='bar', color='skyblue')
    plt.xlabel(groupByVariable)
    plt.ylabel('Revenue')
    plt.title(f'Revenue of Top {topN} {groupByVariable}s')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.ticklabel_format(style='plain', axis='y')
    plt.show()
# %%
#Determine the developer who sold the most units overall and provide a summary of their sales performance:
Display_sales_performance_for_topN_textually('developer_name', 1, dataset_sales, 'Developer', True)

# Convert 'date_sold' column to datetime format
dataset_sales['date_sold'] = pd.to_datetime(dataset_sales['date_sold'])

# Extract year and month from 'date_sold' column
dataset_sales['year'] = dataset_sales['date_sold'].dt.year
dataset_sales['month'] = dataset_sales['date_sold'].dt.month

# Calculate the total price of units sold by each developer
developer_total_price = dataset_sales.groupby('developer_name')['price'].sum()

# Determine the developer who sold the most units overall
top_developer = developer_total_price.idxmax()
# Filter data for the top developer
top_developer_data = dataset_sales[dataset_sales['developer_name'] == top_developer]

# Group data by year and month, then calculate revenue for each group
revenue_by_year_month = top_developer_data.groupby(['year', 'month'])['price'].sum()

# Create a table of revenue by year and month
revenue_table = revenue_by_year_month.reset_index()
revenue_table.columns = ['Year', 'Month', 'Revenue']

# Print the revenue table
print("Revenue by Year and Month for the Top Developer:")
print(revenue_table)

# Create a chart of revenue by year and month
plt.figure(figsize=(12, 6))
plt.plot(revenue_table['Year'].astype(str) + '-' + revenue_table['Month'].astype(str), revenue_table['Revenue'], marker='o', color='skyblue', linestyle='-')
plt.xlabel('Year-Month')
plt.ylabel('Revenue')
plt.title('Revenue by Year and Month for the Top Developer')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.ticklabel_format(style='plain', axis='y')
plt.show()

#%%
# Print top N selling project and number of units they sold
topN = 5
topN_projectCounts = dataset_sales['project_name'].value_counts().head(topN)
project_count_pairs = [f"{category}: {count} units sold" for category, count in topN_projectCounts.items()]
print("Top ", topN , "projects with the highest number of units sold:")
for pair in project_count_pairs:
    print(pair) 

# %%
# Create a visualization that shows the distribution of properties by the number of rooms (layout):
plt.figure(figsize=(10, 6))
sns.countplot(data=dataset_sales, x='layout', hue='layout', legend=False,  palette='pastel')
plt.xlabel('Layout')
plt.ylabel('Number of Properties')
plt.title('Distribution of Properties by Number of Rooms (Layout)')
plt.xticks(rotation=45)
plt.show()

# %%

# %%
# Display the sales performance of the top 5 projects identified in your analysis.
Display_sales_performance_for_topN_textually('project_name', 5, dataset_sales, 'Project', False)

Display_sales_performance_for_topN_inBarChart('project_name', 5, dataset_sales)   # Display top 5 projects graphicaly

# %%
# Print correlation map in order to see which variables correlate to each other. This helps when deciding 
# what variables to use in the model
plt.style.use('ggplot')
corr_dataset = dataset_sales[['price', 'floor_area', 'exterior_area', 'price_p_sqr_meter', 'layout', 'year']]
corr_dataset = pd.get_dummies(corr_dataset, columns=['layout'], drop_first=True)
ext_data_corrs = corr_dataset.corr()

dataPlotColnames = list(ext_data_corrs.columns.values)

plt.figure(figsize = (15, 15))

sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.5, annot = True, vmax = 0.5)
plt.title('Correlation Heatmap')

# From the correlation map, we can assume, that the biggest impact on the price has the floor area. 
print("From the correlation map, we can assume, that the biggest impact on the price has the floor area. ")
# %%
# Conduct a linear regression analysis to determine which factors most significantly impact the price of properties. 
# Consider various property attributes (like floor area or exterior area) in your model.

# Prepare the independent variables (features)
X = dataset_sales[['floor_area', 'exterior_area', 'layout', 'project_name', 'developer_name']]

# Convert categorical variables into dummy/indicator variables
X = pd.get_dummies(X, columns=['layout', 'project_name', 'developer_name'], drop_first=True)

# Prepare the dependent variable
y = dataset_sales['price']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Standardize the features by scaling them
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and fit the HistGradientBoostingRegressor model
model = HistGradientBoostingRegressor()
model.fit(X_train_scaled, y_train)

# Calculate permutation-based feature importances
perm_importance = permutation_importance(model, X_test_scaled, y_test, n_repeats=30, random_state=42)

# Get feature importances from the result
importances = perm_importance.importances_mean

# Create a DataFrame to store feature names and importances
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
top_10_features = feature_importance_df.head(10)
plt.figure(figsize=(10, 6))
plt.barh(top_10_features['Feature'], top_10_features['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Permutation-based Feature Importances')
plt.show()


# Unfortunately, I was not able to perform the linear regression. Instead I used HistGradientBoostingRegressor
# because he handled NaN values that I had problem with in case of pure lin. reg.

print("""Hence we see that variables floor area, layout 3, project 7 and 39 and exterior are are ones of the most 
significant influencers of prices. Naturally, successful project have the top selling units. 
Thanks to that we can see for example which project it is good to invest in.""")
# %%