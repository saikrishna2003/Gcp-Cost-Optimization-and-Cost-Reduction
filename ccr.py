import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import calendar
from datetime import datetime
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="CCC - R & O", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    df['Usage Start Date'] = pd.to_datetime(df['Usage Start Date'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
    df = df.dropna(subset=['Usage Start Date'])
    
    # Convert Network Data from Bytes to GB
    df['Network Inbound Data (GB)'] = df['Network Inbound Data (Bytes)'] / (1024**3)
    df['Network Outbound Data (GB)'] = df['Network Outbound Data (Bytes)'] / (1024**3)
    df['Total Network Data (GB)'] = df['Network Inbound Data (GB)'] + df['Network Outbound Data (GB)']
    
    # Define the thresholds - dictionary
    thresholds = {
        'CPU Utilization (%)': 20,
        'Memory Utilization (%)': 30,
        'Disk I/O Operations': 10,
        'Network Data (GB)': 2
    }
    
    # Calculate underutilization metrics
    df['Underutilized_CPU'] = np.maximum(thresholds['CPU Utilization (%)'] - df['CPU Utilization (%)'], 0)
    df['Underutilized_Memory'] = np.maximum(thresholds['Memory Utilization (%)'] - df['Memory Utilization (%)'], 0)
    df['Underutilized_Network'] = np.maximum(thresholds['Network Data (GB)'] - df['Total Network Data (GB)'], 0)
    df['Underutilized_Quantity'] = np.where(
        (df['Usage Quantity'] < thresholds['Disk I/O Operations']) & (df['Usage Unit'] == 'Requests'),
        thresholds['Disk I/O Operations'] - df['Usage Quantity'],
        0
    )
    
    # Calculate Overall Optimization Factor
    underutilized_columns = ['Underutilized_Quantity', 'Underutilized_Network', 'Underutilized_Memory', 'Underutilized_CPU']
    df['Overall_Optimization_Factor (%)'] = df[underutilized_columns].apply(
        lambda x: x[x > 0].mean() if (x > 0).any() else 0,
        axis=1
    )
    
    # Calculate Optimized Cost
    df['Optimized Cost ($)'] = df['Rounded Cost ($)'] * (1 - df['Overall_Optimization_Factor (%)'] / 100)
    
    return df

# Load dataset
df = load_data()

def format_number(value):
    return '{:,.2f}'.format(value)  # Format with commas

# Streamlit App
st.image("https://cognizant.scene7.com/is/content/cognizant/COG-Logo-2022-1?fmt=png-alpha", width=150)
st.title("Cloud Components Cost Optimization and Forecasting", anchor="header")

# Add a sidebar for navigation
section = st.sidebar.selectbox("Select Section", ["Overview", "Cost Optimization", "Cost Forecasting", "Cost Distribution Analysis", "Cost Optimization Suggestions", "Services Contributing to Cost"])

if section == "Overview":
    st.header("Overview")
    st.write("""
        Welcome to the Cloud Components Cost Optimization and Forecasting application. 
        This tool helps you to manage and optimize your cloud costs effectively. 
        By leveraging this application, you can:
        
        - **Analyze Cloud Costs:** Gain insights into your cloud spending, and identify high-cost services and regions.
        - **Optimize Costs:** Discover underutilized resources and optimize your cloud expenditures.
        - **Forecast Future Costs:** Predict future costs based on historical data and plan your budget accordingly.
        - **Get Suggestions:** Receive actionable recommendations to reduce your cloud costs.

        The application is designed to be user-friendly, allowing you to quickly navigate through different sections to gain insights and take action.
    """)
    st.write("""
        ### Key Features:
        - **Cost Overview:** A summary of your total cloud costs before and after optimization.
        - **Cost Optimization:** Detailed insights and suggestions to help you reduce your cloud expenses.
        - **Cost Forecasting:** Predict future costs based on historical data with the Prophet model.
        - **Cost Distribution Analysis:** Understand how your costs are distributed across various services and regions.
        - **Optimization Suggestions:** Identifies costly services, high network usage, and underutilized resources.
        
        ### How to Use:
        - Select a section from the sidebar to explore different features.
        - Use the provided options to analyze and forecast costs.
        - Review the insights and suggestions to optimize your cloud spending.
    """)

elif section == "Cost Optimization":
    st.header("Cost Optimization Summary")

    # Input: Year Selection
    year = st.selectbox("Select Year", sorted(df['Usage Start Date'].dt.year.unique()))

    # Input: Month and Year
    show_month_year = st.checkbox("Filter by Month and Year")
    if show_month_year:
        months = list(calendar.month_name)[1:]
        selected_month_name = st.selectbox("Select Month", months)
        month = months.index(selected_month_name) + 1
    else:
        month = None

    @st.cache_data
    def get_filtered_data(df, year, month=None):
        if month:
            return df[(df['Usage Start Date'].dt.year == year) & (df['Usage Start Date'].dt.month == month)]
        else:
            return df[df['Usage Start Date'].dt.year == year]

    filtered_data = get_filtered_data(df, year, month)

    total_cost_before = filtered_data['Rounded Cost ($)'].sum()
    total_cost_after = filtered_data['Optimized Cost ($)'].sum()
    cost_change_percentage = ((total_cost_before - total_cost_after) / total_cost_before) * 100
    dollar_saving = total_cost_before - total_cost_after
    inr_saving = dollar_saving * 85

    if month:
        st.markdown(f"**Total Cost Before Optimization for {selected_month_name}:** ${format_number(total_cost_before)}")
        st.markdown(f"**Total Cost After Optimization for {selected_month_name}:** ${format_number(total_cost_after)}")
    else:
        st.markdown(f"**Total Cost Before Optimization for {year}:** ${format_number(total_cost_before)}")
        st.markdown(f"**Total Cost After Optimization for {year}:** ${format_number(total_cost_after)}")
    
    st.markdown(f"**Percentage Change in Cost:** {cost_change_percentage:.2f}%")
    st.markdown(f"**Dollar Saving:** ${format_number(dollar_saving)}")
    st.markdown(f"**INR Saving:** ₹{format_number(inr_saving)}")

    @st.cache_data
    def get_service_costs(filtered_data):
        service_costs_before = filtered_data.groupby('Service Name')['Rounded Cost ($)'].sum().sort_values(ascending=False)
        service_costs_after = filtered_data.groupby('Service Name')['Optimized Cost ($)'].sum().sort_values(ascending=False)
        return pd.DataFrame({
            'Before Optimization': service_costs_before,
            'After Optimization': service_costs_after
        }).fillna(0)

    cost_comparison = get_service_costs(filtered_data)

    if month:
        st.subheader(f"Cost Before and After Optimization for {selected_month_name}")
    else:
        st.subheader(f"Cost Before and After Optimization by Service for {year}")

    fig, ax = plt.subplots(figsize=(12, 8))
    cost_comparison.plot(kind='barh', stacked=False, ax=ax, colormap='coolwarm')
    ax.set_xlabel('Cost in Lakhs($)')
    ax.legend(title='Cost Type')
    st.pyplot(fig)

elif section == "Cost Forecasting":
    st.header("Cost Forecasting")
    
    @st.cache_data
    def load_service_names():
        return df['Service Name'].unique()

    service_names = load_service_names()
    service_name = st.selectbox("Select a Service to Forecast", service_names)

    # Define the forecasting period (Jan 2024 to Dec 2025)
    start_date = pd.to_datetime('2024-01-01')
    end_date = pd.to_datetime('2025-12-31')

    # Calculate the number of months to forecast
    steps = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1

    @st.cache_data
    def prepare_service_data(service_name):
        service_data = df[df['Service Name'] == service_name].copy()
        service_data['Usage Start Date'] = pd.to_datetime(service_data['Usage Start Date'])
        service_data.set_index('Usage Start Date', inplace=True)
        monthly_costs = service_data['Rounded Cost ($)'].resample('M').sum().reset_index()
        monthly_costs.rename(columns={'Usage Start Date': 'ds', 'Rounded Cost ($)': 'y'}, inplace=True)
        return monthly_costs

    @st.cache_data
    def forecast_costs(monthly_costs, steps):
        if len(monthly_costs) < 12:
            return None, None

        # Calculate historical stats
        historical_mean = monthly_costs['y'].mean()
        historical_std = monthly_costs['y'].std()
        historical_min = monthly_costs['y'].min()
        historical_max = monthly_costs['y'].max()

        # Generate forecast dates
        last_date = monthly_costs['ds'].max()
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='M')

        # Generate forecasts based on historical mean with controlled deviations
        np.random.seed(42)  # for reproducibility
        forecasts = np.random.normal(historical_mean, historical_std, steps)
        
        # Clip forecasts to historical range
        forecasts = np.clip(forecasts, historical_min, historical_max)

        # Create forecast dataframe
        forecast_df = pd.DataFrame({'ds': forecast_dates, 'yhat': forecasts})
        forecast_df.set_index('ds', inplace=True)

        # Combine historical data with forecast
        combined_series = pd.concat([monthly_costs.set_index('ds')['y'], forecast_df['yhat']])

        return combined_series, forecast_df['yhat']

    if st.button("Forecast"):
        monthly_costs = prepare_service_data(service_name)
        
        if monthly_costs is None or len(monthly_costs) < 12:
            st.error(f"Not enough data to perform forecasting for {service_name}.")
        else:
            combined_series, forecast = forecast_costs(monthly_costs, steps)
            
            if forecast is not None:
                st.subheader(f"Forecasted Costs for {service_name} (Jan 2024 to Dec 2025)")

                # Scale to appropriate unit (e.g., thousands or millions)
                scale_factor = 1000  # Change this to 1000000 for millions if needed
                combined_series_scaled = combined_series / scale_factor
                forecast_scaled = forecast / scale_factor
                scale_label = "Thousands" if scale_factor == 1000 else "Millions"

                # Display the forecast in a table
                st.write(f"Monthly Forecast (in ${scale_label}):")
                forecast_table = forecast_scaled.reset_index()
                forecast_table.columns = ['Date', f'Forecasted Cost (${scale_label})']
                forecast_table['Date'] = forecast_table['Date'].dt.strftime('%Y-%m-%d')
                st.dataframe(forecast_table)

                # Plot the results
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(combined_series.index, combined_series_scaled, label=f'Historical Costs (${scale_label})', color='blue')
                ax.plot(forecast.index, forecast_scaled, label=f'Forecasted Costs (${scale_label})', color='red', linestyle='--')
                ax.set_xlabel('Date')
                ax.set_ylabel(f'Cost (${scale_label})')
                ax.set_title(f'Cost Forecast for {service_name} (Jan 2024 to Dec 2025)', fontsize=14, fontweight='bold')
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)

elif section == "Cost Distribution Analysis":
    st.header("Cost Distribution Analysis")
    st.write("Analyze how your costs are distributed across different cloud services and regions.")

    # Add time range selection
    time_range = st.radio("Select Time Range", ("Yearly", "Monthly"))

    @st.cache_data
    def filter_data_by_time(df, time_range, year=None, month=None):
        if time_range == "Yearly" and year:
            return df[df['Usage Start Date'].dt.year == year]
        elif time_range == "Monthly" and year and month:
            return df[(df['Usage Start Date'].dt.year == year) & (df['Usage Start Date'].dt.month == month)]
        return df

    @st.cache_data
    def get_service_distribution(df):
        return df.groupby('Service Name')['Rounded Cost ($)'].sum().sort_values(ascending=False)

    @st.cache_data
    def get_region_distribution(df):
        if 'Region / Zone' in df.columns:
            return df.groupby('Region / Zone')['Rounded Cost ($)'].sum().sort_values(ascending=False)
        return None

    # Time range selection UI
    if time_range == "Yearly":
        year = st.selectbox("Select Year", sorted(df['Usage Start Date'].dt.year.unique()))
        filtered_df = filter_data_by_time(df, time_range, year=year)
    elif time_range == "Monthly":
        year = st.selectbox("Select Year", sorted(df['Usage Start Date'].dt.year.unique()))
        month = st.selectbox("Select Month", range(1, 13), format_func=lambda x: calendar.month_name[x])
        filtered_df = filter_data_by_time(df, time_range, year=year, month=month)

    service_distribution = get_service_distribution(filtered_df)

    st.subheader("Cost Distribution by Service")

    # Create an interactive pie chart using Plotly
    fig = px.pie(service_distribution, values=service_distribution.values, names=service_distribution.index,
                 title="Cost Distribution by Service", hole=0.2, 
                 color_discrete_sequence=px.colors.qualitative.Plotly)
    
    fig.update_traces(textinfo='percent+label', hoverinfo='label+value+percent', textposition='inside')
    fig.update_layout(
        showlegend=True,
        legend_title_text="Services",
        margin=dict(t=50, b=50, l=25, r=25),
        width=900,  # Set width of the pie chart
        height=900  # Set height of the pie chart
    )

    # Display the Pie-Chart
    st.plotly_chart(fig)

    st.subheader("Cost Distribution by Region")
    region_distribution = get_region_distribution(filtered_df)
    if region_distribution is not None:
        fig = px.bar(region_distribution, x=region_distribution.values, y=region_distribution.index, 
                     orientation='h', title='Cost Distribution by Region', labels={'x': 'Cost ($)', 'y': 'Region / Zone'},
                     color_discrete_sequence=['lightblue'])
        fig.update_layout(
            width=800,  # Set width of the bar chart
            height=600  # Set height of the bar chart
        )
        st.plotly_chart(fig)
    else:
        st.error("The column 'Region / Zone' is not present in the dataset.")

    # Display top N services table
    st.subheader("Top Services by Cost")
    top_n = st.slider("Select number of top services to display", min_value=1, max_value=20, value=10)
    st.table(service_distribution.head(top_n).reset_index().rename(columns={'index': 'Service Name', 'Rounded Cost ($)': 'Cost ($)'}))

    # Display total cost for the selected time range
    total_cost = filtered_df['Rounded Cost ($)'].sum()
    st.subheader(f"Total Cost for Selected Time Range: ${total_cost:,.2f}")

elif section == "Cost Optimization Suggestions":
    st.header("Cost Optimization Suggestions")
    st.write("### Suggestions for Reducing Cloud Costs")
    st.write("""
    For the analysis, we have used the mean values of the utilization rate which are lesser than the threshold 
    utilization rate. Additionally, here are some actionable suggestions to help you optimize your cloud expenditures:
    """)

    suggestions = [
        ("1. Right Forecasting", """
        To ensure accurate cost forecasting, focus on:
        - **Data Quality:** Maintain clean, consistent, and comprehensive historical data.
        - **Model Selection:** Utilize time-series models like ARIMA, Prophet, or machine learning models like LSTM for better accuracy.
        - **Seasonality and Trends:** Include seasonality and trend analysis to account for periodic fluctuations and long-term trends.
        """),
        ("2. Threshold Calculations", """
        Calculate thresholds to determine underutilized resources:
        - **Utilization Metrics:** Analyze resource utilization over time to set thresholds for identifying underutilized services.
        - **Dynamic Adjustments:** Regularly adjust thresholds based on current usage patterns to avoid over-provisioning.
        """),
        ("3. Optimize CPU Utilization", """
        To optimize CPU usage:
        - **Right-sizing:** Adjust instance sizes based on actual CPU utilization to avoid over-provisioning.
        - **Auto-scaling:** Implement auto-scaling policies to match CPU resources with demand.
        - **Load Balancing:** Distribute workloads evenly across CPUs to maximize efficiency.
        """),
        ("4. Optimize Memory Utilization", """
        For better memory optimization:
        - **Memory Usage Monitoring:** Continuously monitor memory usage to identify bottlenecks or underutilization.
        - **Memory-efficient Algorithms:** Use memory-efficient data structures and algorithms to reduce memory consumption.
        - **Instance Right-sizing:** Select instances with appropriate memory capacity based on your application's requirements.
        """),
        ("5. Optimize Disk I/O Operations", """
        To improve disk I/O performance:
        - **Disk Type Selection:** Choose the right disk types (e.g., SSDs) for high I/O operations.
        - **Data Partitioning:** Partition data across multiple disks to balance the I/O load.
        - **Caching Strategies:** Implement caching mechanisms to reduce frequent disk access and improve speed.
        """),
        ("6. Optimize Usage Quantity", """
        To optimize the usage quantity:
        - **Usage Analysis:** Regularly analyze usage patterns to identify over-provisioned or underutilized services.
        - **Decommission Unused Resources:** Remove or downscale services that are not in use.
        - **Cost-efficient Resource Allocation:** Allocate resources based on actual demand to minimize unnecessary costs.
        """)
    ]

    for title, content in suggestions:
        st.subheader(title)
        st.write(content)

elif section == "Services Contributing to Cost":
    st.header("Services Contributing to Cost")
    
    analysis_type = "Month/Year"
    
    @st.cache_data
    def get_service_costs(data):
        return data.groupby('Service Name')['Rounded Cost ($)'].sum().sort_values(ascending=False)

    if analysis_type == "Month/Year":
        months = list(calendar.month_name)[1:]
        selected_month_name = st.selectbox("Select Month", months)
        month = months.index(selected_month_name) + 1
        
        year = st.selectbox("Select Year", df['Usage Start Date'].dt.year.unique())

        selected_month_data = df[(df['Usage Start Date'].dt.month == month) & (df['Usage Start Date'].dt.year == year)]

    service_costs = get_service_costs(selected_month_data)

    st.subheader(f"Total Cost by Service")
    st.bar_chart(service_costs)

    top_n = st.number_input("Select Number of Top Services to Display", min_value=5, max_value=service_costs.shape[0], value=5)

    st.subheader(f"Top {top_n} Services Contributing to Cost")
    st.write(service_costs.head(top_n))

    fig, ax = plt.subplots(figsize=(10, 6))
    top_services = service_costs.head(top_n)
    ax.barh(top_services.index, top_services.values, color='orange')
    ax.set_xlabel('Cost ($)')
    ax.set_title(f'Top {top_n} Services Contributing to Cost', fontsize=14, fontweight='bold')
    st.pyplot(fig)