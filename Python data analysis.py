# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from statsmodels.stats.proportion import proportions_ztest
import scikit_posthocs as sp
from scipy import stats

"""Connecting to a database, using a query to retrieve information"""

#from google.colab import auth
#from google.cloud import bigquery
#auth.authenticate_user()
#client = bigquery.Client(project="data-analytics-mate")

# query =
"""
SELECT
      s.date,
      s.ga_session_id,
      sp.continent,
      sp.country,
      sp.device,
      sp.browser,
      sp.mobile_model_name AS model_name,
      sp.operating_system,
      sp.language,
      sp.name AS channel_source,
      sp.channel AS channel_traffic,
      acc.id AS customer_id,
      acc.is_verified,
      acc.is_unsubscribed,
      p.category,
      p.name AS product_name,
      p.price,
      p.short_description
FROM `data-analytics-mate.DA.session` AS s
LEFT JOIN `data-analytics-mate.DA.order` AS o
ON s.ga_session_id = o.ga_session_id
LEFT JOIN `data-analytics-mate.DA.product` AS p
ON o.item_id = p.item_id
INNER JOIN `data-analytics-mate.DA.session_params` AS sp
ON s.ga_session_id = sp.ga_session_id
LEFT JOIN `data-analytics-mate.DA.account_session` AS acs
ON s.ga_session_id = acs.ga_session_id
LEFT JOIN `data-analytics-mate.DA.account` AS acc
ON acc.id = acs.account_id
"""

"""Load data with SQL"""

#query_job = client.query(query)  # Executing an SQL query
#results = query_job.result()  # Waiting for the request to complete
#df = results.to_dataframe()

"""On this stage

We connected the necessary modules, connected to the database, executed a query to the database

Load data with CSV file
"""

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive
drive.mount("/content/drive")
# %cd /content/drive/MyDrive
df = pd.read_csv("bq_res_SQL_session.csv")

df['date'] = pd.to_datetime(df['date']) #convert to a correct date from dbdate format
df.head()

"""Checking data in a table"""

# 1. total number of columns and  rows
print(f'Colums total: {df.shape[1]}, Rows total: {df.shape[0]}\n')

# 2. number of numeric columns (and which ones)
numeric_cols = df.select_dtypes(include=np.number)
num_numeric_cols = numeric_cols.shape[1]
print(f"Count colums number type: {num_numeric_cols}")
print(f"Number colums: {list(numeric_cols.columns)}\n")

# 3. number of categorical columns (and which ones)
categorical_cols = df.select_dtypes(include=['object', 'category'])
num_categorical_cols = categorical_cols.shape[1]
print(f"Count colums category type: {num_categorical_cols}")
print(f"Category type colums: {list(categorical_cols.columns)}\n")

# 4. number of datetime columns
datetime_cols = df.select_dtypes(include='datetime')
num_datetime_cols = datetime_cols.shape[1]
print(f"Counts colums date type: {num_datetime_cols}")
print(f"Colums type datetime: {list(datetime_cols.columns)}\n")

# 5. number of unique sessions
unique_sessions = df['ga_session_id'].nunique()
print(f"Counts unique sessions: {unique_sessions}\n")
# 6. what time period is considered (from… to…)
min_time = df["date"].min().date()
max_time = df["date"].max().date()
print(f"The time period considered is: from {min_time} to {max_time}\n")

# 7. Are there any missing values ​​and which columns have more?
missing_values = df.isnull().sum()
total_missing = missing_values.sum()
print(f"Is have empty data: {'Yes' if total_missing > 0 else 'No'}\n")
if total_missing > 0:
    print("Missing values ​​by column:")
    missing_cols_df = missing_values[missing_values > 0].sort_values(ascending=False)
    print(missing_cols_df)

# 8. Which columns have more missing values, what is the reason?
print("\nPossible reasons for missing values:")
print(" - The user did not place an order.")
print(" - For some reason, it is not possible to retrieve data from the browser.")
print(" - There may be errors when reading or storing information.")
print(" - User did not provide information.")

"""What we see, 18 colums and 349545 rows

The time period considered is: from 2020-11-01 to 2021-01-31

some data is not filled in

On which continents (top 3) and in which countries (top 5) does our company have the largest sales? The largest number of orders?
"""

# On which continents (top 3) have the largest sales? The largest number of orders?
df_sales = df.dropna(subset=['price']).copy()
df_sales_cont = df_sales[['continent', 'price']]

# Group by Continent and calculate Revenue for each continent
df_sales_cont_sum = df_sales_cont.groupby('continent')['price'].sum().sort_values(ascending=False)
print(f"Top-3 continent by Revenue \n{df_sales_cont_sum.head(3).to_string(header=False, name=False)} \n")

# Group by Continent and calculate orders for each continent
df_sales_cont_count = df_sales_cont.groupby('continent')['price'].count().sort_values(ascending=False)
print(f"Top-3 continent by Orders \n{df_sales_cont_count.head(3).to_string(header=False, name=False)} \n")

#which countries (top 5) does our company have the largest sales? The largest number of orders?
df_sales_country = df_sales[['country', 'price']]

# Group by Country and calculate Revenue for each country
df_sales_country_sum = df_sales_country.groupby('country')['price'].sum().sort_values(ascending=False)
df_sales_country_sum.index.name = None
print(f"Top-5 country by Revenue \n{df_sales_country_sum.head(5)} \n")

# Group by Country and calculate Orders for each country
df_sales_country_count = df_sales_country.groupby('country')['price'].count().sort_values(ascending=False)
print(f"Top-5 country by Orders \n{df_sales_country_count.head(5).to_string(header=False, name=False)} \n")

"""Summary:

The highest rates by Continent - America for each metrics (Revenue and Orders)

The highest rates by Country - United States for each metrics (Revenue and Orders)

Top 10 product categories by total sales

Top 10 product categories in the country with the highest sales, is the situation different from the general situation?
"""

# top 10 product categories by total sales
df_category = df_sales[['category', 'price']]
df_sales_cont_sum = df_category.groupby('category')['price'].sum().sort_values(ascending=False)
print(f"Top-10 category by Revenue \n{df_sales_cont_sum.head(10).to_string(header=False, name=False)} \n")

# top 10 product categories in the country with the highest sales, or is there a conflict with the overall one?
df_category_top_country = df_sales[['category', 'price', 'country']]
df_category_top_country= df_category_top_country[df_category_top_country['country'] == "United States"]
df_sales_cont_sum = df_category_top_country.groupby('category')['price'].sum().sort_values(ascending=False)
print(f"Top-10 category in United States \n{df_sales_cont_sum.head(10).to_string(header=False, name=False)} \n")
print(f"The top 10 categories in the US and around the world are occupied by similar categories.")

"""Sales by device types and models (as a % of total sales)"""

total_amount = df_sales['price'].sum()
print(f"Total revenue: {total_amount:.2f}\n")

#Group by 'device', 'model_name', clean and sorted data
df_device_total = df_sales[['device', 'model_name', 'price']]
df_device_sales = df_device_total.groupby(['device', 'model_name'])['price'].sum().sort_values(ascending=False)
df_device_prcn_total = pd.DataFrame(df_device_sales)
df_device_prcn_total["percent_of_total"] = (df_device_prcn_total['price'] / total_amount * 100).round(2)
df_device_prcn_total = df_device_prcn_total.reset_index()
df_sorted_custom_order = df_device_prcn_total.sort_values(
    by=['device', 'price'],
    ascending=[True, False]
)
print(f"Revenue and percent of total by devices \n{df_sorted_custom_order.to_string(index=False)} \n")

"""Sales from mobile and desktop have similar values

sales by traffic sources (as a % of total sales)
"""

total_amount = df_sales['price'].sum()
print(f"Total revenue: {total_amount:.2f}\n")

#Group by 'channel_traffic', clean and sorted data
df_channel_traffic_total = df_sales[['channel_traffic', 'price']]
df_channel_traffic_sales = df_channel_traffic_total.groupby('channel_traffic')['price'].sum().sort_values(ascending=False)
df_channel_traffic_prcn_total = pd.DataFrame(df_channel_traffic_sales)
df_channel_traffic_prcn_total["percent_of_total"] = (df_channel_traffic_prcn_total['price'] / total_amount * 100).round(2)
df_channel_traffic_prcn_total = df_channel_traffic_prcn_total.reset_index()

print(f"Revenue and percent of total by devices\n \n{df_channel_traffic_prcn_total.to_string(index=False)} \n")

"""Percentage of registered users confirmed their email address?

What percentage of registered users unsubscribed from the newsletter?

Which countries have the most registered users?

Is there a difference in the behavior (in terms of sales) between those who unsubscribed from the newsletter and those who are still subscribed?
"""

# Create customers data, clean and sort data
df_customer = df.dropna(subset=["customer_id"]).copy()
df_customer = df_customer.reset_index()
# Total amount users
df_customer_amount = len(df_customer["customer_id"].unique())

# Amount verify users
df_customer_amount_verify = df_customer[df_customer['is_verified'] == 1]
verify_customer_amount = df_customer_amount_verify["customer_id"].count()
prcnt_verify_customer = (verify_customer_amount / df_customer_amount * 100).round(2)
print(f"Percentage of registered users confirmed their email address: {prcnt_verify_customer} %\n")

# Amount unsubscribed users
df_customer_amount_subscribed = df_customer[df_customer['is_unsubscribed'] == 1]
subscribed_customer_amount = df_customer_amount_subscribed["customer_id"].count()
prcnt_subscribed_customer = (subscribed_customer_amount / df_customer_amount * 100).round(2)
print(f"Percentage of registered users unsubscribed from the newsletter: {prcnt_subscribed_customer} %\n")

# Amount users by country
df_customer_country = df_customer.groupby("country")["customer_id"].count().sort_values(ascending=False).reset_index()
df_customer_country["percent_of_total_users"] = (df_customer_country['customer_id'] / df_customer_amount * 100).round(2)
df_customer_country = df_customer_country.rename(columns={"customer_id": "amount_users"})
print(f"Countries have the most registered users:\n {df_customer_country.head(10).to_string(index=False)} \n")


df_customer_sales = df_customer.groupby("is_unsubscribed")["price"].describe().reset_index()
df_customer_sales_revenue = df_customer.groupby("is_unsubscribed")["price"].sum().reset_index()
df_customer_sales["revenue"] = df_customer_sales_revenue["price"]
print(df_customer_sales.head().to_string(index=False))

"""71.7 percent have confirmed  their email, most users are from the  United States, order data between users has similar indicators

Create chart for Registered customers by country
"""

df_cst = df_customer_country.head(10) # Limit to 10 countries by count Customers
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=df_cst,
    x="country",
    y="amount_users",
    ax=ax,
    color="skyblue",
    errorbar=('ci', 95)
             )

ax.set_title("Registered customers by country", fontsize=13)
ax.set_ylabel("Registered customers", fontsize=10)
ax.set_xlabel("Country", fontsize=10)
ax.tick_params(axis='y', labelsize=8)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
plt.tight_layout()
plt.show()

"""# Sales dynamics analysis

Total sales for each date and Continent
"""

# Revenue by date table
df_sales_summary = df_sales.groupby(df_sales["date"].dt.to_period("D"))["price"].sum().reset_index()
df_sales_summary['date'] = df_sales_summary['date'].dt.to_timestamp()
# Revenu by Continent table
df_sales_continent = df_sales.groupby([df_sales["date"].dt.to_period("D"), "continent"])["price"].sum().reset_index()
df_sales_continent['date'] = df_sales_continent['date'].dt.to_timestamp()
df_sales_continent_high = df_sales_continent[df_sales_continent['continent'].isin(["Europe", "Asia", "Americas"])]
df_sales_continent_low = df_sales_continent[df_sales_continent['continent'].isin(["Africa", "Oceania", "(not set)"])]
df_sales_continent_low.head()

plt.style.use("ggplot")

fig, ax = plt.subplots(3, 1, figsize=(18, 9))
sns.lineplot(data=df_sales_summary,
    x="date",
    y="price",
    ax=ax[0],
    color='blue',
    marker='o',
    markersize=4,
    linewidth=1.5,
    errorbar=('ci', 95)
             )

ax[0].set_title("Revenue by date", fontsize=13)
ax[0].set_ylabel("Revenue ($)", fontsize=10)
ax[0].set_xlabel("Date", fontsize=10)
ax[0].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
ax[0].tick_params(axis='y', labelsize=8)
ax[0].xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.setp(ax[0].get_xticklabels(), rotation=45, ha='right', fontsize=8)




sns.lineplot(data=df_sales_continent_low,
    x="date",
    y="price",
    ax=ax[1],
    color='blue',
    marker='o',
    markersize=4,
    linewidth=1.5,
    hue="continent"
             )


ax[1].set_title("Revenue by Continent", fontsize=13)
ax[1].set_ylabel("Revenue ($)", fontsize=10)
ax[1].set_xlabel("Date", fontsize=10)
ax[1].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
ax[1].tick_params(axis='y', labelsize=8)
ax[1].xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.setp(ax[1].get_xticklabels(), rotation=45, ha='right', fontsize=8)


sns.lineplot(data=df_sales_continent_high,
    x="date",
    y="price",
    ax=ax[2],
    color='blue',
    marker='o',
    markersize=4,
    linewidth=1.5,
    hue="continent"
             )

ax[2].set_title("Revenue by Continent", fontsize=13)
ax[2].set_ylabel("Revenue ($)", fontsize=10)
ax[2].set_xlabel("Date", fontsize=10)
ax[2].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
ax[2].tick_params(axis='y', labelsize=8)
ax[2].xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.setp(ax[2].get_xticklabels(), rotation=45, ha='right', fontsize=8)


plt.tight_layout()
plt.show()

"""sales dynamics by traffic channel, displays different channels

sales dynamics by device type, displays different devices
"""

# Revenue by Channel Traffic table
df_sales_channel = df_sales.groupby([df_sales["date"].dt.to_period("D"), "channel_traffic"])["price"].sum().reset_index()
df_sales_channel['date'] = df_sales_channel['date'].dt.to_timestamp()


plt.style.use("ggplot")


# Revenue by Channel Traffic Plot
fig, ax = plt.subplots(figsize=(20, 8))
sns.lineplot(data=df_sales_channel,
    x="date",
    y="price",
    marker='o',
    markersize=4,
    ax=ax,
    linewidth=1.5,
    hue="channel_traffic"
             )

ax.set_title("Revenue by Channel", fontsize=13)
ax.set_ylabel("Revenue ($)", fontsize=10)
ax.set_xlabel("Date", fontsize=10)
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
ax.tick_params(axis='y', labelsize=8)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.legend(title="Chaneel Traffic")
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

plt.tight_layout()
plt.show()

# Revenue by Device table
df_sales_device = df_sales.groupby([df_sales["date"].dt.to_period("D"), "device"])["price"].sum().reset_index()
df_sales_device['date'] = df_sales_device['date'].dt.to_timestamp()

#Seasons revenue by Devices
df_seasons = df_sales.groupby([df_sales["date"].dt.day_name(), "device"])["price"].sum().reset_index()
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Create a categorical data type with a defined order
day_of_week_type = pd.CategoricalDtype(categories=days_of_week, ordered=True)

# Apply this type to the 'Day of the week' column
df_seasons['date'] = df_seasons['date'].astype(day_of_week_type)

plt.style.use("ggplot")
fig, ax = plt.subplots(2, 1, figsize=(18, 10))

#Revenue by Device
sns.lineplot(data=df_sales_device,
    x="date",
    y="price",
    ax=ax[0],
    color='blue',
    marker='o',
    markersize=4,
    linewidth=1.5,
    hue="device"
             )

ax[0].set_title("Revenue by Device", fontsize=13)
ax[0].set_ylabel("Revenue ($)", fontsize=10)
ax[0].set_xlabel("Date", fontsize=10)
ax[0].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
ax[0].tick_params(axis='y', labelsize=8)
ax[0].xaxis.set_major_locator(mdates.DayLocator(interval=3))
plt.setp(ax[0].get_xticklabels(), rotation=45, ha='right', fontsize=8)

#Seasons revenue by Devices
sns.lineplot(data=df_seasons,
    x="date",
    y="price",
    ax=ax[1],
    color='blue',
    marker='o',
    markersize=4,
    linewidth=1.5,
    hue="device"
             )

ax[1].set_title("Seasons revenue by Device", fontsize=13)
ax[1].set_ylabel("Revenue ($)", fontsize=10)
ax[1].set_xlabel("Day of Week", fontsize=10)
ax[1].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
ax[1].tick_params(axis='y', labelsize=8)
ax[1].xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.setp(ax[1].get_xticklabels(), rotation=45, ha='right', fontsize=8)

plt.tight_layout()
plt.show()

"""Summary:
  - positive trend
  - Popular devices is Desktop and mobile
  - biggest sales in America, good sales in Europe and Asia also positive trend
  - Popular channel Direct, Organic search, Paid search.
  - there are high abnormal rises and falls, but short-lived
  - There is seasonality, with higher profits on Tuesday and Wednesday, then declining

Pivot tables

A pivot table showing the number of sessions by traffic channels and device types (if the value of a field is unknown, ignore it)

A pivot table showing total sales by product category (top 10 categories) in different countries (top 5 countries)

A pivot table showing total session by devices in different countries
"""

#Pivot table for device and channel
print('Table for device and channel:')
pivot_session_device = pd.pivot_table(data=df, values="ga_session_id", aggfunc="count", index="channel_traffic", columns="device")
display(pivot_session_device)
print('\n')
# Top 10 category
top_10_categories = df_sales.groupby('category')['price'].sum().nlargest(10).index.tolist()

# Top 5 country
top_5_countries = df_sales.groupby('country')['price'].sum().nlargest(5).index.tolist()

# To filter list
df_filtered = df_sales[
    df_sales['category'].isin(top_10_categories) &
    df_sales['country'].isin(top_5_countries)].copy()

#Creat table from filter
pivot_table_sales = pd.pivot_table(
    df_filtered,
    index='category',
    columns='country',
    values='price',
    aggfunc='sum',
    fill_value=0
)
print('Sales category by country:')
display(pivot_table_sales)
print('\n')

# Session count by Country
pivot_session_country = pd.pivot_table(
    df_filtered,
    index='device',
    columns='country',
    values='ga_session_id',
    aggfunc='count',
    fill_value=0
)
print('Sales sessions by country:')
display(pivot_session_country)
print('\n')

# Cteat table median orders by Device
pivot_table_sales_device = pd.pivot_table(
    df_sales,
    index='category',
    columns='device',
    values='price',
    aggfunc='median'
)
print('Sales sales by device:')
display(pivot_table_sales_device)

"""###Statistical analysis of relationships.

- Number of sessions and total sales for each date. Create a visualization to see the relationship between these metrics.
"""

# Total sales for each date
daily_sales = df_sales.groupby('date')['price'].sum().reset_index()
daily_sales.rename(columns={'price': 'total_daily_sales'}, inplace=True)

# 2. Number of unique sessions for each date
daily_sessions = df.groupby('date')['ga_session_id'].nunique().reset_index()
daily_sessions.rename(columns={'ga_session_id': 'daily_sessions'}, inplace=True)

# Merging two DataFrames by date
df_daily_metrics = pd.merge(daily_sales, daily_sessions, on='date', how='inner')
print("\n")

# Visualization of the relationship
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df_daily_metrics, x='daily_sessions', y='total_daily_sales', alpha=0.7, color="blue")

# Add a regression line to visualize the trend
sns.regplot(data=df_daily_metrics, x='daily_sessions', y='total_daily_sales', scatter=False, color='blue', line_kws={'linestyle': '-', 'linewidth': 1})
plt.title('Relationship between daily sessions and total sales', fontsize=16)
plt.xlabel('Number of daily sessions', fontsize=12)
plt.ylabel('Total daily income ($)', fontsize=12)
ax = plt.gca()
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
plt.grid(True, linestyle='-', alpha=0.6)
plt.tight_layout()
plt.show()

# Calculating Pearson's correlation coefficient and p-value
from scipy.stats import pearsonr
correlation_coefficient, p_value = pearsonr(df_daily_metrics['daily_sessions'], df_daily_metrics['total_daily_sales'])
print("\n")
print(f"Pearson correlation coefficient between daily sessions and total sales: {correlation_coefficient:.4f}")
print(f"P-value: {p_value:.4f}")

# Statistical significance analysis
alpha = 0.05
if p_value < alpha:
    print(f"P-value ({p_value:.4f}) less than the significance level ({alpha}), therefore the correlation is statistically significant.")
else:
    print(f"P-value ({p_value:.4f}) more than the significance level ({alpha}), therefore the correlation is not statistically significant.")

"""Are sales on different continents correlated, top 3 continents. Analysis of statistical significance of relationships"""

total_revenue_by_continent = df_sales.groupby('continent')['price'].sum().nlargest(3)
top_3_continents = total_revenue_by_continent.index.tolist()

df_daily_continent_sales = df_sales.groupby(['date', 'continent'])['price'].sum().reset_index()
df_pivot_sales = df_daily_continent_sales.pivot_table(
    index='date',
    columns='continent',
    values='price'
)

df_pivot_sales = df_pivot_sales[top_3_continents]
print("Total sales per day by continent:")
display(df_pivot_sales.head())
print("\n")

# Calculating the Pearson correlation matrix
print("Corelation matrix by continent:")
correlation_matrix = df_pivot_sales.corr(method='pearson')
display(correlation_matrix)
print("\n")

# Correlation matrix visualization (Heatmap)
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",
            linewidths=.5, cbar_kws={'label': 'Correlation coefficients'})
plt.title('Correlation matrix dayli sales from Continent', fontsize=14)
plt.show()


# Matrix and P-value
p_value_matrix = pd.DataFrame(np.zeros((len(top_3_continents), len(top_3_continents))),
                              columns=top_3_continents, index=top_3_continents)
significance_summary = []

alpha = 0.05

for i in range(len(top_3_continents)):
    for j in range(i + 1, len(top_3_continents)):
        continent1 = top_3_continents[i]
        continent2 = top_3_continents[j]

        corr, p_val = pearsonr(df_pivot_sales[continent1], df_pivot_sales[continent2])

        p_value_matrix.loc[continent1, continent2] = p_val
        p_value_matrix.loc[continent2, continent1] = p_val

        status = "statistically significant" if p_val < alpha else "not statistically significant"
        significance_summary.append(
            f"Correlation between {continent1} and {continent2}: R={corr:.4f}, p={p_val:.4f}. It is {status}.")

print("\nP-value matrix:")

with pd.option_context("display.float_format", '{:.9f}'.format):
      display(p_value_matrix)

print("\nSummary of statistical significance:")
for s in significance_summary:
    print(s)

"""Positive average correlation between continents

Are sales across different traffic channels correlated?

Analyze the statistical significance of the relationships.
"""

df_daily_channel_sales = df_sales.groupby(['date', 'channel_traffic'])['price'].sum().reset_index()
total_revenue_by_continent = df_sales.groupby('channel_traffic')['price'].sum()
top_channels = total_revenue_by_continent.index.tolist()
df_pivot_channel_sales = df_daily_channel_sales.pivot_table(
    index='date',
    columns='channel_traffic',
    values='price',
)

# Calculating the Pearson correlation matrix
correlation_matrix_channels = df_pivot_channel_sales.corr(method='pearson')

print("Correlation matrix by channel:")
display(correlation_matrix_channels)
print("\n")

# Correlation matrix visualization (Heatmap)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_channels, annot=True, cmap='coolwarm', fmt=".2f",
            linewidths=.5, cbar_kws={'label': 'Correlation coefficient'})
plt.title(f'Correlation matrix of daily sales by traffic channels', fontsize=12)
plt.show()

# Matrix and P-value
p_value_matrix_channels = pd.DataFrame(np.zeros((len(top_channels), len(top_channels))),
                                       columns=top_channels, index=top_channels)
significance_summary_channels = []

alpha = 0.05

for i in range(len(top_channels)):
    for j in range(i + 1, len(top_channels)):
        channel1 = top_channels[i]
        channel2 = top_channels[j]

        corr, p_val = pearsonr(df_pivot_channel_sales[channel1], df_pivot_channel_sales[channel2])

        p_value_matrix_channels.loc[channel1, channel2] = p_val
        p_value_matrix_channels.loc[channel2, channel1] = p_val

        status = "statistically significant" if p_val < alpha else "not statistically significant"
        significance_summary_channels.append(
            f"Correlation between {channel1} and {channel2}: R={corr:.4f}, p={p_val:.4f}. It is {status}."
        )

print("\nP-value matrix:")
with pd.option_context('display.float_format', '{:.6f}'.format):
    display(p_value_matrix_channels)

print("\nSummary of statistical significance:")
for s in significance_summary_channels:
    print(s)

"""Positive low-average correlation between channels

Relationship analysis, comparison of average check by category
"""

#Describe table for visual analysis
print(f"Describe table:")
display(df_sales.groupby('category')['price'].describe())
print(f"\n")
#Creat boxplot for visual analysis
plt.figure(figsize=(15, 6))
sns.set_style("whitegrid")
sns.boxplot(x='category', y='price', data=df_sales, width=0.5, fliersize=3, hue="category")
plt.title('Order price by Category', fontsize=14)
plt.xticks(rotation=20, ha='right', fontsize=7)
plt.yticks(fontsize=7)
plt.xlabel("Caterogy", fontsize=10)
plt.ylabel("Order price", fontsize=10)
plt.tight_layout()
plt.show()

#Checking the number of emissions using the quartile method
print(f"\n")
outlier_summary = []

for category, group in df_sales.groupby('category'):
    Q1 = group['price'].quantile(0.25)
    Q3 = group['price'].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    n_outliers = ((group['price'] < lower) | (group['price'] > upper)).sum()
    total = len(group)
    perc = (n_outliers / total * 100) if total > 0 else 0

    outlier_summary.append({
        'Category': category,
        'Count otrliers': n_outliers,
        'Total count': total,
        '% otrliers': round(perc, 1)
    })

outlier_table = pd.DataFrame(outlier_summary).sort_values('% otrliers', ascending=False).reset_index()
display(outlier_table)
print(f"\n")




# Statistical significance test, method kruskal
# We get a list of arrays, where each array is all the price values ​​for a certain category
from scipy.stats import kruskal
groups = [group['price'].values for name, group in df_sales.groupby('category')]

# Test Kruskal-Wallis
stat, p = kruskal(*groups)

print(f"Statistics H = {stat:.3f}, p-value = {p:.4f}")

if p < 0.05:
    print("There is a statistically significant difference between the categories (medians are different)")
else:
    print("There is no statistically significant difference between categories (medians are similar)")
print("\n")

#Dunn test
dunn_test_results = sp.posthoc_dunn(df_sales, val_col='price', group_col='category', p_adjust='bonferroni')

print("Dunn's test P-value matrix (with Bonferroni correction)")
alpha_test = 0.05
significant_difference_matrix = dunn_test_results < alpha_test

print("Statistically significant differences matrix:")
with pd.option_context('display.float_format', '{:.6f}'.format):
    display(significant_difference_matrix)
print(f"\n")

#Median price by category (ranking)
median_prices = df_sales.groupby('category')['price'].median().sort_values(ascending=False)
print("Median price by category (ranking):")
display(median_prices)

median_prices = df_sales.groupby('category')['price'].median().sort_values(ascending=False)

low_group_max = median_prices.quantile(0.33)  # 33 percentile
high_group_min = median_prices.quantile(0.75) # 75 percentile

print(f"Base Group Limit (lower 33%): {low_group_max:.2f}")
print(f"Premium group limit (high 67%): {high_group_min:.2f}")

df_medians = median_prices.reset_index(name='median_price')

# Function for defining a group
def assign_group(price):
    if price >= high_group_min:
        return 'High'
    elif price <= low_group_max:
        return 'Low'
    else:
        return 'Medium'

# Add new colum with group
df_medians['Price_Group'] = df_medians['median_price'].apply(assign_group)
print("\nDistribution of categories by groups:")
display(df_medians[['category','median_price', 'Price_Group']])
print("\n")

# Filter data for a specific group
df_group_high = df_medians[df_medians['Price_Group'] == 'High']
df_group_medium = df_medians[df_medians['Price_Group'] == 'Medium']
df_group_low = df_medians[df_medians['Price_Group'] == 'Low']
# Sort data within a group (for better readability)
df_group_high = df_medians[df_medians['Price_Group'] == 'High'].sort_values(by='median_price', ascending=True)
df_group_medium = df_medians[df_medians['Price_Group'] == 'Medium'].sort_values(by='median_price', ascending=True)
df_group_low = df_medians[df_medians['Price_Group'] == 'Low'].sort_values(by='median_price', ascending=True)

# Creat graph
plt.style.use("ggplot")
fig, ax = plt.subplots(3,1, figsize=(11, 11))

y_max_total = median_prices.max()
y_offset_base = y_max_total * 0.1


groupe_cat_objects = {
    "High": (df_group_high, 0),
    "Medium": (df_group_medium, 1),
    "Low": (df_group_low, 2)
}
for group_name, (df_group, n) in groupe_cat_objects.items():
    # Get the axis for the current graph
    axes_group = ax[n]

    # Graph construction
    sns.barplot(
        x='category',
        y='median_price',
        data=df_group,
        ax=axes_group,
        hue='category'
    )

    # Graph settings
    axes_group.set_title(f'Median: {group_name} groupe', fontsize=12, pad=10)
    axes_group.set_ylabel('Median ($)', fontsize=10)
    axes_group.set_xlabel('Category', fontsize=10)
    axes_group.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))

    # Rotate X labels for each graph
    plt.setp(axes_group.get_xticklabels(), rotation=0, ha='center', fontsize=8)
    sorted_categories = df_group['category'].tolist()

    y_offset = y_offset_base

    for i in range(len(sorted_categories)):
        for j in range(i + 1, len(sorted_categories)):
            cat1 = sorted_categories[i]
            cat2 = sorted_categories[j]

            # If the difference is not significant (False)
            if not significant_difference_matrix.loc[cat1, cat2]:

                x1 = i
                x2 = j

                y_val_1 = df_group[df_group['category'] == cat1]['median_price'].values[0]
                y_val_2 = df_group[df_group['category'] == cat2]['median_price'].values[0]

                y_level = max(y_val_1, y_val_2) + y_offset

                axes_group.plot([x1, x1, x2, x2], [y_level - (y_offset*0.1), y_level, y_level, y_level - (y_offset*0.1)], c='black', lw=1.0)

                axes_group.text((x1 + x2) / 2, y_level, "Not Significant", ha='center', va='bottom', fontsize=7, color='black')

                y_offset += y_offset * 0.6


    y_max_current = df_group['median_price'].max()
    axes_group.set_ylim(0, y_max_current + y_offset)

plt.tight_layout()
plt.show()

"""Sales by top 5 product categories correlated

Analyze the statistical significance of the relationships.
"""

total_revenue_by_category = df_sales.groupby('category')['price'].sum().nlargest(5)
top_5_category = total_revenue_by_category.index.tolist()

df_daily_category_sales = df_sales.groupby(['date', 'category'])['price'].sum().reset_index()
df_pivot_sales = df_daily_category_sales.pivot_table(
    index='date',
    columns='category',
    values='price'
)

print("Total sales per day by channel:")
df_pivot_sales = df_pivot_sales[top_5_category]
display(df_pivot_sales.head())

# Calculating the Pearson correlation matrix
print("\nCorrelation matrix by channel:")
correlation_matrix = df_pivot_sales.corr(method='pearson')
display(correlation_matrix)

print("\n")
# Correlation matrix visualization (Heatmap)
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",
            linewidths=.5, cbar_kws={'label': 'Correlation coefficients'})
plt.title('Correlation matrix dayli sales from Category', fontsize=14)
plt.show()


# Matrix and P-value
p_value_matrix = pd.DataFrame(np.zeros((len(top_5_category), len(top_5_category))),
                              columns=top_5_category, index=top_5_category)
significance_summary = []

alpha = 0.05

for i in range(len(top_5_category)):
    for j in range(i + 1, len(top_5_category)):
        category1 = top_5_category[i]
        category2 = top_5_category[j]

        corr, p_val = pearsonr(df_pivot_sales[category1], df_pivot_sales[category2])

        p_value_matrix.loc[category1, category2] = p_val
        p_value_matrix.loc[category2, category1] = p_val

        status = "statistically significant" if p_val < alpha else "not statistically significant"
        significance_summary.append(
            f"Correlation between {category1} and {category2}: R={corr:.4f}, p={p_val:.4f}. It is {status}.")

print("\nP-value matrix:")
with pd.option_context('display.float_format', '{:.9f}'.format):
    display(p_value_matrix)

print("\nSummary of statistical significance:")
for s in significance_summary:
    print(s)

"""###Statistical analysis of differences between groups"""

df_sales['is_registered'] = ~df_sales['customer_id'].isna()

df_daily_sales_by_status = df_sales.groupby(['date', 'is_registered'])['price'].sum().reset_index()

df_daily_pivot = df_daily_sales_by_status.pivot_table(
    index='date',
    columns='is_registered',
    values='price',
    fill_value=0
)

df_daily_pivot.columns = ['Unregistered_Sales', 'Registered_Sales']
df_daily_pivot = df_daily_pivot.fillna(0)

registered_sales_sample = df_daily_pivot['Registered_Sales']
unregistered_sales_sample = df_daily_pivot['Unregistered_Sales']
print("Summary information about Customers \n")
print(f"Registered customers days: {len(registered_sales_sample)}")
print(f"Unregistered customers days: {len(unregistered_sales_sample)}")
print(f"Average daily sales of registered customer: ${registered_sales_sample.mean():,.2f}")
print(f"Average daily sales of unregistered customer: ${unregistered_sales_sample.mean():,.2f}")
print("\n")

#Visualisation
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(registered_sales_sample, kde=True, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of daily sales of registered customers', fontsize=12)
plt.xlabel('Daily sales ($)', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.gca().xaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))

plt.subplot(1, 2, 2)
sns.histplot(unregistered_sales_sample, kde=True, bins=30, color='lightcoral', edgecolor='black')
plt.title('Distribution of daily sales of unregistered customers', fontsize=12)
plt.xlabel('Daily sales ($)', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.gca().xaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))

plt.tight_layout()
plt.show()
print("\n")


from scipy import stats
print("Statistics test for Customers group \n")
# Shapiro-Wilk test for registered customers
shapiro_reg = stats.shapiro(registered_sales_sample)
print(f"Shapiro-Wilk test for registered sales:")
print(f"Statistic W: {shapiro_reg.statistic:.4f}")
print(f"P-value:   {shapiro_reg.pvalue:.4f} \n")

# Shapiro-Wilk test for unregistered customers
shapiro_unreg = stats.shapiro(unregistered_sales_sample)
print(f"Shapiro-Wilk test for unregistered sales:")
print(f"Statistic W: {shapiro_unreg.statistic:.4f}")
print(f"P-value:   {shapiro_unreg.pvalue:.4f} \n")


alpha_normality = 0.05

if shapiro_reg.pvalue < alpha_normality:
    print(f"Registered customers: P-value ({shapiro_reg.pvalue:.4f}) < {alpha_normality}. The null hypothesis of normality is rejected. The data are not normally distributed.")
else:
    print(f"Registered customers: P-value ({shapiro_reg.pvalue:.4f}) >= {alpha_normality}. There is not enough evidence to reject the null hypothesis. The data may be normally distributed.")

if shapiro_unreg.pvalue < alpha_normality:
    print(f"Unregistered customers: P-value ({shapiro_unreg.pvalue:.4f}) < {alpha_normality}. The null hypothesis of normality is rejected. The data are not normally distributed.")
else:
    print(f"Unregistered customers: P-value ({shapiro_unreg.pvalue:.4f}) >= {alpha_normality}. There is not enough evidence to reject the null hypothesis. The data may be normally distributed.")

levene_test = stats.levene(registered_sales_sample, unregistered_sales_sample)

print("\n")
print(f"Levene's test for equality of variances:")
print(f"Statistic F: {levene_test.statistic:.4f}")
print(f"P-value:   {levene_test.pvalue:.4f}")
print("\n")

alpha_variance = 0.05

if levene_test.pvalue < alpha_variance:
    print(f"P-value ({levene_test.pvalue:.4f}) < {alpha_variance}. The null hypothesis of equality of variances is rejected. The sample variances are not equal.")
else:
    print(f"P-value ({levene_test.pvalue:.4f}) >= {alpha_variance}. There is insufficient evidence to reject the null hypothesis. The sample variances are assumed to be equal.")

mannwhitney_test = stats.mannwhitneyu(registered_sales_sample, unregistered_sales_sample, alternative='two-sided')

print("\n")
print(f"Mann-Whitney U statistic: {mannwhitney_test.statistic:.4f}")
print(f"P-value:   {mannwhitney_test.pvalue:.4f}")

print(f"\nMedian daily sales:")
print(f"  Registered customers: ${registered_sales_sample.median():,.2f}")
print(f"  Unregistered customers: ${unregistered_sales_sample.median():,.2f}")

print(f"\nAverage daily sales:")
print(f"  Registered customers: ${registered_sales_sample.mean():,.2f}")
print(f"  Unregistered customers: ${unregistered_sales_sample.mean():,.2f}")

alpha_test = 0.05

if mannwhitney_test.pvalue < alpha_test:
    print(f"  Since the P-value ({mannwhitney_test.pvalue:.4f}) less than the significance level ({alpha_test}),")
    print("  we reject the null hypothesis (H0).")
    print("  This means that there is a statistically significant difference in daily sales between registered and unregistered users.")
    if registered_sales_sample.median() > unregistered_sales_sample.median():
        print(f"  Median daily sales of registered users (${registered_sales_sample.median():,.2f}) statistically significantly higher than in unregistered (${unregistered_sales_sample.median():,.2f}).")
    else:
        print(f"  Median daily sales of unregistered users (${unregistered_sales_sample.median():,.2f}) statistically significantly higher than that of registered (${registered_sales_sample.median():,.2f}).")
else:
    print(f"  Since the P-value ({mannwhitney_test.pvalue:.4f}) more than the significance level ({alpha_test}),")
    print("  we do not have sufficient reason to reject the null hypothesis (H0).")
    print("  This means that there is no statistically significant difference in daily sales between registered and unregistered users.")

"""We analyze the number of sessions by channel traffic. Choose the appropriate test. Are there statistically significant differences between the groups?"""

df_daily_channel_sessions = df.groupby(['date', 'channel_traffic'])['ga_session_id'].count().reset_index()
df_pivot_channel_sessions = df_daily_channel_sessions.pivot_table(index='date', columns='channel_traffic', values='ga_session_id', fill_value=0)
channel_groups = df_pivot_channel_sessions.columns.tolist()
data_for_kruskal = [df_pivot_channel_sessions[channel].values for channel in channel_groups]
kruskal_test_result = stats.kruskal(*data_for_kruskal)

print("Kruskal-Wallis test results:")
print(f"  Statistics H: {kruskal_test_result.statistic:.4f}")
print(f"  P-value:   {kruskal_test_result.pvalue:.4f}")

alpha_test = 0.05

if kruskal_test_result.pvalue < alpha_test:
    print(f"  Since the P-value ({kruskal_test_result.pvalue:.4f}) less than the significance level ({alpha_test}),")
    print("  we reject the null hypothesis (H0).")
else:
    print(f"  Since the P-value ({kruskal_test_result.pvalue:.4f}) more than the significance level ({alpha_test}),")
    print("  we do not have sufficient reason to reject the null hypothesis (H0).")

print("\nMedian daily sessions for each channel:")
for channel in channel_groups:
    median_sessions = df_pivot_channel_sessions[channel].median()
    print(f"  Channel '{channel}': Median = {median_sessions:,.2f}")

alpha_test = 0.05


if kruskal_test_result.pvalue < alpha_test:

    dunn_test_results = sp.posthoc_dunn(data_for_kruskal, p_adjust='bonferroni')
    dunn_test_results.columns = channel_groups
    dunn_test_results.index = channel_groups

    print("Dunn's test P-value matrix (with Bonferroni correction):")
    with pd.option_context('display.float_format', '{:.6f}'.format):
        display(dunn_test_results)

    print("\nInterpretation of Dunn's Test results:")
    significant_pairs = []
    for i in range(len(channel_groups)):
        for j in range(i + 1, len(channel_groups)):
            channel1 = channel_groups[i]
            channel2 = channel_groups[j]
            p_value = dunn_test_results.loc[channel1, channel2]

            if p_value < alpha_test:
                significant_pairs.append(f"{channel1} vs {channel2} (p={p_value:.4f})")
                print(f"  > Statistically significant difference: {channel1} and {channel2} (P-value: {p_value:.4f})")

    if not significant_pairs:
        print(" No statistically significant differences were found between any pairs of channels (after correcting for multiple comparisons).")
    else:
        print("\n Post-hoc Analysis Conclusion:")
        print("Statistically significant differences were found in the median of daily sessions between the following pairs of channels:")
        for pair in significant_pairs:
            print(f"- {pair}")

else:
    print("\nPost-hoc analysis was not required because the Kruskal-Wallis test revealed no overall statistically significant difference between channels.")

"""Is the proportion of sessions with organic traffic statistically different in Europe and America?"""

df_daily_channel_sessions_continent = df.groupby(['date','continent', 'channel_traffic'])['ga_session_id'].count().reset_index()
df_continent_filtered_continent = df_daily_channel_sessions_continent[df_daily_channel_sessions_continent['continent'].isin(['Europe', 'Americas'])].copy()
df_continent_filtered_channel = df_continent_filtered_continent[df_continent_filtered_continent["channel_traffic"] == 'Organic Search']

#'Europe', 'Americas'
organic_sessions_europe = df_continent_filtered_channel[
    df_continent_filtered_channel['continent'] == 'Europe'
]['ga_session_id'].sum()

organic_sessions_americas = df_continent_filtered_channel[
    df_continent_filtered_channel['continent'] == 'Americas'
]['ga_session_id'].sum()


total_sessions_europe = df_continent_filtered_continent[
    df_continent_filtered_continent['continent'] == 'Europe'
]['ga_session_id'].sum()

total_sessions_americas = df_continent_filtered_continent[
    df_continent_filtered_continent['continent'] == 'Americas'
]['ga_session_id'].sum()

print("Prepared data for Z-test of proportions:")
print(f"Europe: Total sessions = {total_sessions_europe:,.0f}, Organic sessions = {organic_sessions_europe:,.0f}")
print(f"Americas: Total sessions = {total_sessions_americas:,.0f}, Organic sessions = {organic_sessions_americas:,.0f}")

if total_sessions_europe == 0 or total_sessions_americas == 0:
    print("Error: One or both regions have no common sessions. Unable to perform Z-test.")
elif organic_sessions_europe > total_sessions_europe or organic_sessions_americas > total_sessions_americas:
    print("Error: The number of organic sessions exceeds the total number of sessions for one of the regions. Check the aggregation.")

else:
    count = np.array([organic_sessions_europe, organic_sessions_americas])
    nobs = np.array([total_sessions_europe, total_sessions_americas])

    z_statistic, p_value_ztest = proportions_ztest(count, nobs, alternative='two-sided')

    alpha_ztest = 0.05
    print("\n")
    print("Z-test results for comparing organic traffic shares:")
    print(f"Z-statistic: {z_statistic:.4f}")
    print(f"P-value: {p_value_ztest:.4f}")

    if p_value_ztest < alpha_ztest:
        print(f" Since the P-value ({p_value_ztest:.4f}) is less than the significance level ({alpha_ztest}),")
        print(" we reject the null hypothesis (H0).")
        print(" This means that there is a statistically significant difference in the proportion of sessions with organic traffic between Europe and America.")

        prop_europe = organic_sessions_europe / total_sessions_europe
        prop_americas = organic_sessions_americas / total_sessions_americas
        if prop_europe > prop_americas:
            print(f" (The share of organic traffic in Europe ({prop_europe:.2%}) is statistically higher than in America ({prop_americas:.2%}).)")
        else:
            print(f" (The share of organic traffic in America ({prop_americas:.2%}) is statistically higher than in Europe ({prop_europe:.2%}).)")
    else:
        print(f" Since the P-value ({p_value_ztest:.4f}) is greater than or equal to the significance level ({alpha_ztest}),")
        print(" we have insufficient evidence to reject the null hypothesis (H0).")
        print(" This means that **there is no statistically significant difference** in the proportion of sessions with organic traffic between Europe and America.")

"""Tableau project link

https://public.tableau.com/app/profile/pasha.b5021/viz/Project1_17560525065480/Dashboard1?publish=yes
"""
