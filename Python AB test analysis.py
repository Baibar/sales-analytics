# -*- coding: utf-8 -*-


Import modules
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from statsmodels.stats.proportion import proportions_ztest
import warnings

"""SQL query"""

# query =
'''
with session_info as(
select
        s.date,
        s.ga_session_id,
        sp.country,
        sp.device,
        sp.continent,
        sp.channel,
        ab.test,
        ab.test_group
from `data-analytics-mate.DA.ab_test` as ab
join `data-analytics-mate.DA.session` as s
on s.ga_session_id = ab.ga_session_id
join `data-analytics-mate.DA.session_params` as sp
on sp.ga_session_id = ab.ga_session_id
),
session_with_orders as (
select
        session_info.date,
        session_info.country,
        session_info.device,
        session_info.continent,
        session_info.channel,
        session_info.test,
        session_info.test_group,
        count(distinct o.ga_session_id) as session_with_orders
from `data-analytics-mate.DA.order` as o
join session_info
on o.ga_session_id = session_info.ga_session_id
group by
        session_info.date,
        session_info.country,
        session_info.device,
        session_info.continent,
        session_info.channel,
        session_info.test,
        session_info.test_group),


events as (
select
        session_info.date,
        session_info.country,
        session_info.device,
        session_info.continent,
        session_info.channel,
        session_info.test,
        session_info.test_group,
        sp.event_name,
        count(sp.ga_session_id) as event_cnt
from `data-analytics-mate.DA.event_params` as sp
join session_info
on sp.ga_session_id = session_info.ga_session_id
group by
        session_info.date,
        session_info.country,
        session_info.device,
        session_info.continent,
        session_info.channel,
        session_info.test,
        session_info.test_group,
        sp.event_name),


session as (
select
        session_info.date,
        session_info.country,
        session_info.device,
        session_info.continent,
        session_info.channel,
        session_info.test,
        session_info.test_group,
        count(distinct session_info.ga_session_id) as session_cnt
from session_info
group by
        session_info.date,
        session_info.country,
        session_info.device,
        session_info.continent,
        session_info.channel,
        session_info.test,
        session_info.test_group),


account as(
select
        session_info.date,
        session_info.country,
        session_info.device,
        session_info.continent,
        session_info.channel,
        session_info.test,
        session_info.test_group,
        count(distinct acs.ga_session_id) as new_account_cnt
from `data-analytics-mate.DA.account_session` as acs
join session_info
on acs.ga_session_id = session_info.ga_session_id
group by
        session_info.date,
        session_info.country,
        session_info.device,
        session_info.continent,
        session_info.channel,
        session_info.test,
        session_info.test_group)


select
        session_with_orders.date,
        session_with_orders.country,
        session_with_orders.device,
        session_with_orders.continent,
        session_with_orders.channel,
        session_with_orders.test,
        session_with_orders.test_group,
        'session with orders' as event_name,
        session_with_orders.session_with_orders as value
from `session_with_orders`
union all
select
        events.date,
        events.country,
        events.device,
        events.continent,
        events.channel,
        events.test,
        events.test_group,
        events.event_name,
        events.event_cnt as value
from `events`
union all
select
        session.date,
        session.country,
        session.device,
        session.continent,
        session.channel,
        session.test,
        session.test_group,
        "session" as event_name,
        session.session_cnt as value
from `session`
union all
select
        account.date,
        account.country,
        account.device,
        account.continent,
        account.channel,
        account.test,
        account.test_group,
        "new account" as event_name,
        new_account_cnt as value
from `account`
''

#from google.colab import auth
#from google.cloud import bigquery
#auth.authenticate_user()
#client = bigquery.Client(project="data-analytics-mate")
#query_job = client.query(query)  # Executing an SQL query
#results = query_job.result()  # Waiting for the request to complete
#df = results.to_dataframe()

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive
drive.mount("/content/drive")
# %cd /content/drive/MyDrive
df = pd.read_csv("bq-results-2025.csv")

# Convert Date to correct format
df['date'] = pd.to_datetime(df['date'])
# Review the first data
df.head(5)

"""General information about the dataset"""

# total number of columns and  rows
print(f'Colums total: {df.shape[1]}, Rows total: {df.shape[0]}\n')

# number of numeric columns (and which ones)
numeric_cols = df.select_dtypes(include=np.number)
num_numeric_cols = numeric_cols.shape[1]
print(f"Count colums number type: {num_numeric_cols}")
print(f"Number colums: {list(numeric_cols.columns)}\n")

# number of categorical columns (and which ones)
categorical_cols = df.select_dtypes(include=['object', 'category'])
num_categorical_cols = categorical_cols.shape[1]
print(f"Count colums category type: {num_categorical_cols}")
print(f"Category type colums: {list(categorical_cols.columns)}\n")

# number of datetime columns
datetime_cols = df.select_dtypes(include='datetime')
num_datetime_cols = datetime_cols.shape[1]
print(f"Counts colums date type: {num_datetime_cols}")
print(f"Colums type datetime: {list(datetime_cols.columns)}\n")

# what time period is considered (from… to…)
min_time = df["date"].min().date()
max_time = df["date"].max().date()
print(f"The time period considered is: from {min_time} to {max_time}\n")

# Are there any missing values ​​and which columns have more?
missing_values = df.isnull().sum()
total_missing = missing_values.sum()
print(f"Is have empty data: {'Yes' if total_missing > 0 else 'No'}\n")
if total_missing > 0:
    print("Missing values ​​by column:")
    missing_cols_df = missing_values[missing_values > 0].sort_values(ascending=False)
    print(missing_cols_df)

"""Set the initial settings for viewing tests, metrics, and data

"""

metric_list = ['view_promotion', 'session with orders', 'new account', 'add_shipping_info', 'add_payment_info', 'add_to_cart']
test_data = ['test', 'test_group', 'event_name']

"""Creating a function for automated testing, accepts two lists for test setup - metrics, test_data

metrics - should include the metrics we are interested in from the Event Name column, except Session

test_data - must include columns
- test,
- test_group,
- event_name

To test_data can also add additional columns for analysis from the list
- country,
- device,
- continent,
- channel
"""

def zp_stats_value(metrics, test_data):
  print(f"List of data for test {test_data}\n")
  print(f"List of metrics for test {metrics}\n")
  # Form a table with the specified metrics
  event_counts = (
      df[df['event_name'].isin(metrics)]
      .groupby(test_data)['value']
      .sum()
      .reset_index())
  # Form a table with the specified session for data
  session_counts = df.copy()
  session_counts = session_counts[session_counts['event_name'] == 'session']

  no_evn = test_data.copy()
  no_evn.remove('event_name')
  session_counts = session_counts.groupby(no_evn)['value'].sum().reset_index()

  # Merged both tables for result
  merged = event_counts.merge(session_counts, on=(no_evn), how='left')

  for_index_data = test_data.copy()
  for_index_data.remove('test_group')

  # Create pivot tables
  pivoted = merged.pivot_table(
      index= (for_index_data),
      columns='test_group',
      values=['value_x', 'value_y'],
      fill_value=0)
  # Expand the table for correct display
  pivoted.columns = [f"{col[0]}_group_{col[1]}" for col in pivoted.columns]
  pivoted = pivoted.reset_index()

  col_numb = len(test_data)
  # calculate converion rate for each group
  pivoted['conversion_group_1'] = pivoted[pivoted.columns[col_numb - 1]] / pivoted[pivoted.columns[col_numb + 1]]
  pivoted['conversion_group_2'] = pivoted[pivoted.columns[col_numb]] / pivoted[pivoted.columns[col_numb + 2]]
  # Rename colums
  pivoted.columns = [*for_index_data,
                'event_count_group_1','event_count_group_2', 'session_count_group_1', 'session_count_group_2',
                'conversion_group_1', 'conversion_group_2']

  # Organize colums
  cols_table = [*for_index_data,
                'event_count_group_1','session_count_group_1','conversion_group_1',
                'event_count_group_2', 'session_count_group_2', 'conversion_group_2']
  pivoted = pivoted[cols_table]
  # calculate converion change for each group
  pivoted['conversion_change_%'] = (pivoted['conversion_group_2'] - pivoted['conversion_group_1']) / pivoted['conversion_group_1'] * 100

  table_1 = pivoted.copy()
  # Add the necessary columns for test results
  table_1['z_stat'] = None
  table_1['p_value'] = None
  table_1['significant'] = None
  # Set the limit of a statistically significant result 5%
  x = 0.05
  for i, row in table_1.iterrows():

    successes = [int(row['event_count_group_1']), int(row['event_count_group_2'])]
    nobs = [int(row['session_count_group_1']), int(row['session_count_group_2'])]


    if any(s <= 0 for s in successes) or any(n <= 0 for n in nobs):
        z_stat, p_val, statis = None, None, 'NA'  # або 'FALSE' чи інше
    else:
         with warnings.catch_warnings():
              warnings.simplefilter("ignore")
              z_stat, p_val = proportions_ztest(successes, nobs)
              statis = 'TRUE' if p_val < x else 'FALSE'
    # Set result test to the table
    table_1.loc[i, 'z_stat'] = z_stat
    table_1.loc[i, 'p_value'] = p_val
    table_1.loc[i, 'significant'] = statis
  return table_1

"""Notes: we are comparing group 1 to group 2

Interpretation of results:

Z stat value less than 0 - means that group 2 has better results than group 1

Z stat value greater than 0 - means that group 2 has worse results than group 1

Check statistical data according to the general scheme
"""

test_data = ['test', 'test_group', 'event_name']
event_check_test = zp_stats_value(metric_list, test_data)
display(event_check_test)

"""Statistical test by continent"""

continent_data = ['test', 'test_group', 'event_name', 'continent']
continent_check_test = zp_stats_value(metric_list, continent_data)
display(continent_check_test)

"""Statistical test by device"""

device_data = ['test', 'test_group', 'event_name', 'device']
device_check_test = zp_stats_value(metric_list, device_data)
display(device_check_test)

"""Statistical test by channel"""

channel_data = ['test', 'test_group', 'event_name', 'channel']
channel_check_test = zp_stats_value(metric_list, channel_data)
display(channel_check_test)

"""Export results"""

#event_check_test.to_csv('event_check_test.csv', index=False)
#channel_check_test.to_csv('channel_check_test.csv', index=False)
#device_check_test.to_csv('device_check_test.csv', index=False)
#continent_check_test.to_csv('continent_check_test.csv', index=False)

"""Tableau AB Test [link](https://public.tableau.com/app/profile/pasha.b5021/viz/ABtestexample_17517160561460/ABtest#1)
