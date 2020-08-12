import pandas as pd

data_path = "/dbfs/mnt/databricks-datasets-private/ML/GlobalMobility"
womply_raw_pd = pd.read_csv(data_path + "/womply_revenue.csv")
states_pd = pd.read_csv(data_path + "/states.csv")

baseline = 100000
joined_pd = womply_raw_pd.join(states_pd, on="statefips", rsuffix="state")
joined_pd['date'] = (joined_pd['year'].map(str) + "-" + joined_pd['month'].map(str) + "-" + joined_pd['day'].map(str)).astype('datetime64')
joined_pd = joined_pd[['date', 'statename', 'revenue_all']].copy()
joined_pd = joined_pd.rename(columns={'statename': 'state'})
joined_pd = joined_pd[joined_pd['date'] >= '2020-02-04']
joined_pd['revenue'] = (joined_pd['revenue_all'] + 1) * baseline
forecast_pd = joined_pd.drop('revenue_all', axis=1)

print(forecast_pd.sort_values(["state", "date"]))


monthly_growth = 0.01

forecast_pd['month'] = pd.DatetimeIndex(forecast_pd['date']).month
forecast_pd = forecast_pd.groupby(['state', 'month']).mean()
forecast_pd = forecast_pd.sort_values(['state', 'month'])
forecast_pd['forecast'] = forecast_pd.groupby(['state']).shift(1) * (1 + monthly_growth)
forecast_pd = forecast_pd.dropna()
forecast_pd['mape'] = (forecast_pd['revenue'] - forecast_pd['forecast']).abs() / forecast_pd['revenue']

print(forecast_pd)
