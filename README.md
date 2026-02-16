# Data Analytics Project: E-commerce Performance & A/B Testing Analysis

This project provides a comprehensive analysis of the UrbanScape Apparel marketplace performance. The study is divided into two key phases: strategic business metrics analysis and A/B testing for conversion rate optimization.

## 📊 Interactive Dashboards
You can explore the interactive visualizations and key performance indicators for this project here:

[Link to Tableau Dashboard: Business Overview](https://public.tableau.com/app/profile/pasha.b5021/viz/Project1_17560525065480/Dashboard1)

[Link to Tableau Dashboard: A/B Testing Results](https://public.tableau.com/app/profile/pasha.b5021/viz/ABtestexample_17517160561460/ABtest)

## 1. Business Recommendations
* Scale the US Market: Since the USA generates the majority of revenue, localized marketing campaigns should be implemented to strengthen market share.

* Product Category Optimization: Increase marketing budget for "Sofas & Armchairs" and "Chairs," as these categories show the highest consumer demand.

* Prioritize Mobile Experience: A significant portion of traffic comes from iPhone/Safari users; therefore, mobile UI/UX improvements should be a top priority.

* Implement A/B Test Winners: Based on statistically significant results (p-value < 0.05), it is recommended to deploy the changes from the test group (Group 2) globally to improve the "Add Payment Info" conversion.

* Retention Strategy: With a 16.94% unsubscribed rate, the email marketing strategy needs revision to improve long-term customer retention.

## 2. Key Insights & Summary
* Sustainable Growth: The business shows a healthy sales dynamic with a clear correlation between traffic volume (sessions) and total revenue.

* Channel Efficiency: Organic Search is the most powerful sales driver, confirming the effectiveness of the current SEO strategy.

* Experiment Results: A/B testing confirmed that interface/process modifications in the test group successfully increased the probability of purchase completion.

* Technology Stack: Users prefer modern browsers (Chrome, Safari), which simplifies technical platform support.

## 3. Key Results
* Total Revenue: Over $17.66M during the study period (Nov 2020 – Jan 2021).

* Market Leader: The USA accounts for approximately $13.94M of total income.

* Conversion: The highest activity at the add_payment_info stage is observed in Social Search and Undefined channels (up to 8%).

* Email Loyalty: 71.7% of registered users verified their email addresses.

* Statistical Correlation: A direct link between user activity and daily revenue was confirmed via Pearson correlation (p-value < 0.05).

## 4. Methodology: Data Collection, Analysis & Visualization

### Tech Stack
1. SQL: BigQuery (Joins, CTEs, Aggregations)
2. Python: Pandas, NumPy
3. Statistics: SciPy (Pearson Correlation), Statsmodels (Z-test for Proportions)

### Visualization: Tableau, Matplotlib, Seaborn

### Data Processing

1. Sourcing: Data was extracted from Google BigQuery.
2. Cleaning: Used Pandas for data wrangling, handling missing values, and datetime transformations.
3. SQL Engineering: Joined, using optimized SQL queries.
4. Statistical Analysis & Visualization
5. Descriptive Statistics: Segmented data by geography, product categories, and device types.
6. Hypothesis Testing: Applied Z-test for Proportions to evaluate A/B test groups.
7. Correlation Analysis: Identified key drivers of revenue using scipy.stats.
8. Visuals: Developed a series of time-series charts, conversion funnels, and heatmaps to communicate insights effectively.
