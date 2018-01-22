# Brought along the changes we made to the `dc_listings` Dataframe.
dc_listings = pd.read_csv('dc_airbnb.csv')
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')
dc_listings = dc_listings.loc[np.random.permutation(len(dc_listings))]
np.random.seed(1)

def predict_price(new_listing):
    temp_df = dc_listings.copy()
    ## Complete the function.іі
    temp_df['distance'] = temp_df['accommodates'].apply(lambda x: np.abs(x - new_listing))
    temp_df = temp_df.sort_values('distance')

    new_listing = temp_df.iloc[:5]['price'].mean() 
    return(new_listing)

acc_one = predict_price(1)
print(acc_one)
acc_two = predict_price(2)
print(acc_two)
acc_four = predict_price(4)
print(acc_four)