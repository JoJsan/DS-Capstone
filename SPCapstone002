import pandas as pd
import streamlit as st
from scipy import stats
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
 

# In[12]:


# Markdown content
st.title("Mount Verlud Data Exploration (Data Scientist Capstone)")
markdown_content = """
**by Joseph James Santiago**
## Introduction

In the quest to ensure access to clean and safe drinking water, alternative sources play a crucial role. This exploration centers on Mount Verlud, a local water refilling station, aiming to analyze sales and expense data for operational efficiency.

## Background and Rationale:

Mount Verlud, situated in Sariaya, Quezon serves as a vital source of purified water for the local community. Understanding sales patterns and expense management is essential for optimizing operations and ensuring a reliable supply of safe drinking water. The project leverages data analytics to enhance efficiency, with a focus on Mount Verlud's specific needs.

In the Philippines, water refilling stations have gained popularity due to their affordability and convenience. The project aims to harness the power of data analytics to gain insights into Mount Verlud's sales trends and expense patterns.

## Objective:

The project's overarching objective is to leverage data analytics for enhancing operational efficiency and strategic decision-making at Mount Verlud. Specific objectives include creating a sales dashboard, implementing an anomalous expense detection system, developing a demand forecasting model, and conducting regression analysis between gross sales and product categories.

## Problem Statement:

Mount Verlud faces challenges such as a lack of real-time sales monitoring, difficulty in identifying expense anomalies, unpredictability in water demand, and a limited understanding of the contribution of each product category to overall gross sales. The project aims to address these challenges through the implementation of data-driven solutions, empowering Mount Verlud with actionable insights for efficiency and strategic growth.

## Glossary of Terms:

1. **Date:**
   - The specific date of the recorded sales and expense data.

2. **Gross Sales:**
   - The total revenue generated from sales before deducting any expenses.

3. **Expenses:**
   - The total costs incurred by Mount Verlud, including driver and helper wages, gas cost for the delivery truck, cost for filters, labels, bottles, seals, water supply bill, Hydro/ROI (Return on Investment), and NBE (Non-Business Related Expense).

4. **Product Categories:**
   - Specific water container types offered by Mount Verlud, including Slim (4L), Round (4L), 1L Bottle, 500ml Bottle, and 350ml Bottle.

5. **Slim (4L):**
   - The rectangular container with a 4-liter capacity.

6. **Round (4L):**
   - The round container with a 4-liter capacity.

7. **1L Bottle:**
   - A bottle with a 1-liter capacity.

8. **500ml Bottle:**
   - A bottle with a 500ml capacity.

9. **350ml Bottle:**
   - A bottle with a 350ml capacity.

## Methodologies:

The data for analysis was sourced from the Store Manager's sales and expense Excel data, covering the months from June to October. Python, Streamlit, and Plotly were employed for analysis and visualization, with the implementation of the ARIMA model for demand forecasting. The methodologies ensure a comprehensive and technologically advanced approach to analyzing Mount Verlud's sales, expenses, and demand patterns.

## Limitations:

1. **Inconsistent Pricing:**
   - Mount Verlud faced challenges in establishing a fixed price per product category, leading to inconsistent pricing. Some customers were sold certain products at different prices. As a result, pricing data lacks uniformity, restricting its inclusion in the analysis. The focus remains on reliable metrics such as quantity sold per product and Gross Sales.

2. **Limited Information on Customer Base:**
   - The dataset lacks comprehensive information on the customer base, including demographics, preferences, and purchase behavior. This limitation hinders a detailed customer-centric analysis, restricting insights into customer segmentation and targeted marketing strategies.

3. **Real-time Sales Monitoring:**
   - Mount Verlud currently lacks real-time sales monitoring capabilities. The analysis is based on historical data, and real-time insights into sales trends are not available.

4. **Expense Anomalies Identification:**
   - While the project aims to implement an anomalous expense detection system, the effectiveness of identifying and addressing expense anomalies is contingent on the completeness and accuracy of the dataset.

5. **Unpredictability in Water Demand:**
   - The ARIMA model is employed for demand forecasting; however, water demand can be influenced by external factors that may not be fully captured by historical data. Unforeseen events or changes in local circumstances could impact demand unpredictability.

6. **Limited Historical Data Period:**
   - The dataset covers sales and expense data for the months from June to October. The relatively short historical period may limit the robustness of certain analyses, especially those sensitive to seasonal trends or long-term patterns.

Acknowledging these limitations is crucial for interpreting the study's findings and recognizing potential areas for improvement in future data collection and analysis efforts. These limitations provide context for the scope and reliability of the insights derived from the data science capstone project at Mount Verlud.
"""

# Render Markdown content
st.markdown(markdown_content)



# In[13]:


results_markdown =""""
## Results and Discussion
Using pandas we load the consolidated dataset from the water refilling station."""

st.markdown(results_markdown)


# In[14]:


# Load the consolidated file
consolidated = pd.read_excel(r'C:\Users\Ryzen 5\Documents\Capstone\consolidated.xlsx')


# In[6]:


consolidated.head()


# In[7]:


consolidated.dtypes


# In[9]:


consolidated.describe()


# In[16]:


import plotly.express as px


# In[18]:


# Metrics section with a box and dropdown for months
st.header("Sales and Expense dashboard")
st.subheader("Metrics")

# Dropdown for selecting months
month_options = ['June', 'July', 'August', 'September', 'October', 'Total']
selected_month = st.selectbox("Select Month", month_options)

# Filter data by selected month
if selected_month == 'Total':
    filtered_by_month = consolidated
else:
    filtered_by_month = consolidated[consolidated['Date'].dt.strftime('%B') == selected_month]

# Calculate Total Expense and Total Sales for the selected month
total_expense_selected_month = filtered_by_month['Total Expense'].sum()
total_sales_selected_month = filtered_by_month['Gross Sales'].sum()

# Calculate Net Profit and Percentage Profit
net_profit_selected_month = total_sales_selected_month - total_expense_selected_month
percentage_profit_selected_month = (net_profit_selected_month / total_sales_selected_month) * 100 if total_sales_selected_month != 0 else 0

# Display the metric box for Total Expense
st.markdown(f'<div style="border: 2px solid red; padding: 10px; color: red;">'
            f'<p>Total Expense: ₱{total_expense_selected_month:,.2f}</p></div>', unsafe_allow_html=True)

# Display the metric box for Total Sales
st.markdown(f'<div style="border: 2px solid blue; padding: 10px; color: blue;">'
            f'<p>Total Sales: ₱{total_sales_selected_month:,.2f}</p></div>', unsafe_allow_html=True)

# Display the metric box for Net Profit and Percentage Profit
st.markdown(f'<div style="border: 2px solid green; padding: 10px; color: green;">'
            f'<p>Net Profit: ₱{net_profit_selected_month:,.2f} '
            f'({percentage_profit_selected_month:.2f}%)</p></div>', unsafe_allow_html=True)

st.write("This simple dashboard summarizes the breakdown of sales and expenses for the each month.")

# Doughnut chart for Sales breakdown
st.subheader("Sales Breakdown")
sales_columns = ["Slim", "Round", "Total 350", "Total 500", "Total 1L"]
sales_data = filtered_by_month[sales_columns].sum()
fig_sales = px.pie(sales_data, values=sales_data, names=sales_columns, hole=0.3)
st.plotly_chart(fig_sales)

# Doughnut chart for Expense breakdown
st.subheader("Expense Breakdown")
expense_columns = ["Gas", "Driver", "Helper", "Bottles", "Label", "Seal", "Filters", "Water Delivery", "Hydro/ROI", "NBE"]
expense_data = filtered_by_month[expense_columns].sum()
fig_expense = px.pie(expense_data, values=expense_data, names=expense_columns, hole=0.3)
st.plotly_chart(fig_expense)



# Line graph for Gross Sales and Total Expense with additional filtering options
st.subheader("Gross Sales and Total Expense Over Time")

# Dropdown for selecting months
line_graph_month_options = ['June', 'July', 'August', 'September', 'October', 'Total']
selected_line_graph_month = st.selectbox("Select Month for Line Graph", line_graph_month_options)

# Filter data by selected month for line graph
if selected_line_graph_month == 'Total':
    line_graph_filtered_data = consolidated
else:
    line_graph_filtered_data = consolidated[consolidated['Date'].dt.strftime('%B') == selected_line_graph_month]

# Resample data based on the selected time granularity
resample_freq = 'D'  # Daily frequency for line graph
resampled_line_graph_data = line_graph_filtered_data.set_index('Date').resample(resample_freq).sum()

# Line chart for Gross Sales and Total Expense
fig_line_time = px.line(resampled_line_graph_data, x=resampled_line_graph_data.index,
                        y=['Gross Sales', 'Total Expense'], title='Gross Sales and Total Expense Over Time')

# Set color for Gross Sales and Total Expense
fig_line_time.update_traces(line=dict(color='blue'), selector=dict(name='Gross Sales'))
fig_line_time.update_traces(line=dict(color='red'), selector=dict(name='Total Expense'))

# Show the line chart
st.plotly_chart(fig_line_time)

# Bar chart for Sales by Day of the Week
st.subheader("Sales by Day of the Week")

# Dropdown for selecting months
month_options_bar = ['June', 'July', 'August', 'September', 'October', 'Total']
selected_month_bar = st.selectbox("Select Month for Bar Chart", month_options_bar)

# Filter data by selected month for the bar chart
if selected_month_bar == 'Total':
    filtered_by_month_bar = consolidated
else:
    filtered_by_month_bar = consolidated[consolidated['Date'].dt.strftime('%B') == selected_month_bar]

# Group data by day of the week and sum the sales
sales_by_day = filtered_by_month_bar.groupby(filtered_by_month_bar['Date'].dt.day_name())['Gross Sales'].sum()

# Arrange the days chronologically
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sales_by_day = sales_by_day.reindex(days_order)

# Bar chart for Sales by Day of the Week
fig_bar = px.bar(sales_by_day, x=sales_by_day.index, y=sales_by_day.values, labels={'x': 'Day of the Week', 'y': 'Total Sales'}, color=sales_by_day.values, color_continuous_scale='Blues')
st.plotly_chart(fig_bar)

st.write("Above is a chart showing cumulative sales per day. We can see which day of each has the most sales")

# Anomaly Detection - Z-test for Total Expense
# Calculate z-scores for Total Expense
consolidated['z_score_expense'] = (consolidated['Total Expense'] - consolidated['Total Expense'].mean()) / consolidated['Total Expense'].std()

# Set a threshold for anomaly detection (adjust as needed)
z_score_threshold = 2.5

# Identify anomalous expenses based on the z-scores
anomalous_expenses = consolidated[consolidated['z_score_expense'].abs() > z_score_threshold]

# Visualization of Total Expenses Over Time with Anomalous Dates Highlighted
st.subheader("Total Expenses Over Time with Anomalous Dates")

# Line chart for Total Expenses
fig_total_expenses = px.line(consolidated, x='Date', y='Total Expense', title='Total Expenses Over Time')

# Highlight anomalous dates with a different color
fig_total_expenses.add_trace(go.Scatter(x=anomalous_expenses['Date'], y=anomalous_expenses['Total Expense'],
                                        mode='markers', name='Anomalous Expenses',
                                        marker=dict(color='green', size=8, line=dict(color='black', width=2))))

# Set color for Total Expenses line
fig_total_expenses.update_traces(line=dict(color='red'))

# Show the line chart
st.plotly_chart(fig_total_expenses)

st.write("This Anomlay expenses detector detected 3 anomalous expense. However, it may not be anomalous at all as the spike in expense coincided with a sudden bulk order in those dates which also resulted in a sales spike.")

# Regression Analysis
st.subheader("Regression Analysis")
st.write("A simple regression Analysis to see how each product category contribute to Gross Sales")

# Dropdown for selecting product category
product_options = ['Slim', 'Round', 'Total 350', 'Total 500', 'Total 1L']
selected_product = st.selectbox("Select Product Category", product_options)

# Select data for the chosen product category
regression_data = consolidated[[selected_product, 'Gross Sales']]

# Drop rows with missing values
regression_data = regression_data.dropna()

# Define independent variable (X) and dependent variable (y)
X = sm.add_constant(regression_data[selected_product])  # Add a constant term to the independent variable
y = regression_data['Gross Sales']

# Fit the regression model
try:
    model = sm.OLS(y, X).fit()
    # Display the regression summary
    st.write(model.summary())

    # Plotting the regression line
    fig_regression = px.scatter(x=regression_data[selected_product], y=regression_data['Gross Sales'], labels={selected_product: 'Quantity Sold', 'Gross Sales': 'Sales'},
                                title=f'Regression Plot for {selected_product}')
    
    # Add the regression line to the plot
    fig_regression.add_trace(go.Scatter(x=regression_data[selected_product], y=model.predict(X), mode='lines', name='Regression Line'))

    # Show the plot
    st.plotly_chart(fig_regression)

    # Dynamic text interpretation
    st.subheader(f"Regression Results Interpretation for {selected_product}")
    st.text(f"R-squared: {model.rsquared:.3f}")
    st.text(f"Interpretation: The R-squared value of {model.rsquared:.3f} indicates that the linear regression model "
            f"explains {model.rsquared * 100:.2f}% of the variability in Gross Sales based on the Quantity Sold for {selected_product}. "
            f"The relationship suggests that {model.params[selected_product]:.3f} unit increase in {selected_product} is associated with "
            f"a {model.params[selected_product] * 100:.2f}% increase in Gross Sales.")
except Exception as e:
    # Print exception details for debugging
    st.error(f"Error: {str(e)}")
    st.write("X:", X)
    st.write("y:", y)
    st.write("regression_data:", regression_data)

#Forcase section
# Create an empty DataFrame to store forecasts
waterforecast = pd.DataFrame()

# Iterate over each product category
for selected_product in ['Slim', 'Round', 'Total 350', 'Total 500', 'Total 1L']:
    # Filter data for the selected product
    df_product = consolidated[['Date', selected_product]].copy()
    df_product.set_index('Date', inplace=True)

    # Filter data for the training period (June to October)
    training_data = df_product.loc['2023-06-01':'2023-10-31']

    # Train-test split
    train_size = int(len(training_data) * 0.9)
    train, test = training_data.iloc[:train_size], training_data.iloc[train_size:]

    # Fit ARIMA model
    model = ARIMA(train, order=(5, 1, 0))
    fit_model = model.fit()

    # Evaluate the model on the test data
    predictions = fit_model.forecast(steps=len(test))
    mae = mean_absolute_error(test[selected_product], predictions)
    mse = mean_squared_error(test[selected_product], predictions)
    rmse = np.sqrt(mse)

    # Forecast future quantity sold for November
    forecast_steps = 30  # Assuming 30 days in November
    forecast_index = pd.date_range(start='2023-11-01', periods=forecast_steps, freq='D')
    forecast = fit_model.forecast(steps=forecast_steps)

    # Add forecast to waterforecast DataFrame
    waterforecast[selected_product + '_Forecast'] = forecast

    # Display results for each category
    st.subheader(f'Demand Forecasting for {selected_product} in November')
    fig_forecast = go.Figure()

    # Plot the training data
    fig_forecast.add_trace(go.Scatter(x=train.index, y=train[selected_product], mode='lines', name='Train'))

    # Plot the test data
    fig_forecast.add_trace(go.Scatter(x=test.index, y=test[selected_product], mode='lines', name='Test'))

    # Plot the forecast data for November
    fig_forecast.add_trace(go.Scatter(x=forecast_index, y=forecast, mode='lines', name='Forecast'))

    # Update layout for better readability
    fig_forecast.update_layout(title=f'Demand Forecasting for {selected_product} in November with ARIMA',
                               xaxis_title='Date',
                               yaxis_title='Quantity Sold',
                               legend=dict(x=0, y=1, traceorder='normal'))

    # Show the plot
    st.plotly_chart(fig_forecast)

# Display the waterforecast DataFrame
st.subheader("Water Forecast DataFrame")
st.dataframe(waterforecast)

# Interpretation of Model Evaluation Metrics
st.markdown("""
### My Interpretation of Model Evaluation Metrics

Hey there! Let's dive into what these metrics mean for our demand forecasting:

- **Mean Absolute Error (MAE):**
  - On average, my model's predictions are off by around 13.60 units. This gives me a sense of the typical size of errors in my forecasts.

- **Mean Squared Error (MSE):**
  - The average squared difference between predicted and actual values is approximately 282.83. MSE emphasizes larger errors, providing insight into the overall error magnitude.

- **Root Mean Squared Error (RMSE):**
  - Taking the square root of MSE, I get an RMSE of about 16.82. This is a more user-friendly metric, indicating the typical size of errors in my predictions.

These metrics help me assess how accurate my demand forecasting model is. Generally, lower values are better, but it's essential to consider the specific context of my application. Excited to keep improving and refining the model!
""")


# Capacity of each product in liters
product_capacity = {'Slim': 4, 'Round': 4, 'Total 350': 0.35, 'Total 500': 0.5, 'Total 1L': 1}

# Tank capacity and refill frequency
tank_capacity = 1000  # in liters
refill_frequency = 2  # in days

# Calculate the total forecasted demand for each day
waterforecast['Total_Demand'] = waterforecast.sum(axis=1)

# Calculate the required water supply for each day
waterforecast['Required_Supply'] = waterforecast['Total_Demand'] * refill_frequency

# Check if the available water supply is sufficient
waterforecast['Sufficient_Supply'] = waterforecast['Required_Supply'] <= tank_capacity

# Display the results
st.subheader("Water Supply Analysis for November")
st.dataframe(waterforecast[['Total_Demand', 'Required_Supply', 'Sufficient_Supply']])

# Visualize the results
fig_supply_analysis = go.Figure()

# Plot the total forecasted demand
fig_supply_analysis.add_trace(go.Scatter(x=waterforecast.index, y=waterforecast['Total_Demand'], mode='lines', name='Total Demand'))

# Plot the required water supply
fig_supply_analysis.add_trace(go.Scatter(x=waterforecast.index, y=waterforecast['Required_Supply'], mode='lines', name='Required Supply'))

# Highlight days with sufficient supply
sufficient_days = waterforecast[waterforecast['Sufficient_Supply']].index
fig_supply_analysis.add_trace(go.Scatter(x=sufficient_days, y=waterforecast.loc[sufficient_days, 'Required_Supply'],
                                         mode='markers', marker=dict(color='green'), name='Sufficient Supply'))

# Highlight days with insufficient supply
insufficient_days = waterforecast[~waterforecast['Sufficient_Supply']].index
fig_supply_analysis.add_trace(go.Scatter(x=insufficient_days, y=waterforecast.loc[insufficient_days, 'Required_Supply'],
                                         mode='markers', marker=dict(color='red'), name='Insufficient Supply'))

# Update layout for better readability
fig_supply_analysis.update_layout(title='Water Supply Analysis for November',
                                  xaxis_title='Date',
                                  yaxis_title='Water Quantity (liters)',
                                  legend=dict(x=0, y=1, traceorder='normal'))

# Show the plot
st.plotly_chart(fig_supply_analysis)
st.write("Forecast data shows that the station would have an ample supply of water to meet the demands provided that there won't be an unforeseen event that will lead to water interruption")

# In[ ]:
# Conclusion section
st.subheader("Conclusion")
st.write("In conclusion, we created a simple sales dashboard and saw that the water station appears to be profitable so far."
         "The majority of the expenses go to labor costs of the delivery driver and the helper." "Around 60% of the sales come from the Slim, 350ml, and 500ml containers."
         "We built an expense anomaly detector and in context discovered that the spike in expense doesn't seem anomalous at all as per the store manager those dates coincided with bulk orders from customers."
            "Based on the Regrssion analysis, the 350ml containers had the highest Rsquared of .102 which is still relatively on the low side which means the model still warrants further invesigation."
            "We built a forecast model and determined that the store would have ample water supply to meet the demand forecasts for the month of November")

# Recommendations section
st.subheader("Recommendations")
st.write("We recommend a better data collection practice. Especially, on the customer base side of things so we can gain more insight on that regard."
        "Based on the analysis, it is recommended to monitor the water supply closely, especially if the estimated days between refills indicate potential shortages. "
         "Consider adjusting refill intervals, increasing tank capacity, or exploring alternative water supply solutions to ensure adequate resources for the store.")

# References section
st.subheader("References")

# Reference 1
reference_1 = (
    "1. Magtibay, B. B. (2004). Water refilling station: an alternative source of drinking water supply in the Philippines. "
    "In Proceedings of the 30th WEDC International Conference, Vientiane, Lao PDR. PEOPLE-CENTRED APPROACHES TO WATER AND ENVIRONMENTAL SANITATION."
)
st.write(reference_1)

# Reference 2
reference_2 = (
    "2. Alambatin, A. K. V., Germano, J. C., Pagaspas, D. L., Peñas, F. M. D., Pun-an, A., & Galarpe, V. R. K. R. (2017). "
    "Drinking Water Quality of Selected Tap Water Samples in Cagayan de Oro (District II), Philippines. Journal of Sustainable Development Studies, 10(1), 1-16. "
    "ISSN 2201-4268."
)
st.write(reference_2)


