import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import Polynomial

# Load the dataset
df = pd.read_csv('Crimes in Malaysia.csv')

# Aggregate the total number of crimes per year
df_agg = df.groupby('year')['crimes'].sum().reset_index()

# Prepare the data for polynomial regression
x = df_agg['year'].values
y = df_agg['crimes'].values

# Perform polynomial regression (degree 2)
p = Polynomial.fit(x, y, deg=2)

# Generate the model values for the existing data
mymodel = p(x)

# Predict the total number of crimes for 2024 to 2028
future_years = np.arange(2024, 2029)
future_predictions = p(future_years)

# Append the future predictions to the existing data
all_years = np.concatenate((x, future_years))
all_predictions = np.concatenate((mymodel, future_predictions))

# Plot the original data and the polynomial regression line
plt.scatter(x, y, label='Data (2016-2023)')
plt.plot(all_years, all_predictions, color='red', label='Prediction')
plt.xlabel('Year')
plt.ylabel('Total Crimes')
plt.title('Forecast of Total Crimes Index in 5 Years')
plt.legend()
plt.show()

# Print the predicted total number of crimes for 2024 to 2028
for year, prediction in zip(future_years, future_predictions):
    print(f"Prediction of total amount of crimes in {year}: {round(prediction)}")