import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset into a Pandas DataFrame
data = pd.read_csv('India_Electric_Vehicle_Market.csv')

# Segment the data
# Example: Create a bar chart for market share by region
region_market_share = data['Region'].value_counts()
plt.figure(figsize=(10, 6))
plt.bar(region_market_share.index, region_market_share.values)
plt.xlabel('Region')
plt.ylabel('Market Share (Units)')
plt.title('Market Share of EVs by Region')
plt.show()
