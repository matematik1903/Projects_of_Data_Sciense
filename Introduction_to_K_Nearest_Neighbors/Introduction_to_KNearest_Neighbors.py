# Introduction to the data
import pandas as pd

dc_listings = pd.read_csv('dc_airbnb.csv')
print(dc_listings.loc[0])

# Euclidean distance
import numpy as np

dc_listings = pd.read_csv('dc_airbnb.csv')
first_living_space_value = dc_listings.loc[0]['accommodates']
first_distance = np.abs(first_living_space_value - 3)
print(first_distance)

# Calculate distance for all observations
new_listing = 3
dc_listings['distance'] = dc_listings['accommodates'].apply(lambda x: np.abs(x - new_listing))
print(dc_listings['distance'].value_counts())

# Randomizing, and sorting
import numpy as np
np.random.seed(1)

dc_listings = dc_listings.loc[np.random.permutation(len(dc_listings))]
dc_listings = dc_listings.sort_values('distance')
print(dc_listings.iloc[0:10]["price"])

# Average price
stripped_commas = dc_listings["price"].str.replace('$', '')
stripped_dollars = stripped_commas.str.replace(',', '')

dc_listings['price'] = stripped_dollars.astype('float')
mean_price = dc_listings.iloc[:5]['price'].mean() 

print(mean_price)