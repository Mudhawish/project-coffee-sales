import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import statsmodels.api as sm

# --- 1. CONFIGURATION AND STYLING ---
st.set_page_config(
    page_title="Professional Coffee Sales Dashboard",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# New color palette: a modern, elegant, coffee-inspired scheme
PLOTLY_COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]
ALTAIR_THEME = {
    "config": {
        "title": {"fontSize": 22, "font": "Georgia", "anchor": "middle", "fontWeight": "bold"},
        "axis": {"titleFontSize": 14, "labelFontSize": 12, "grid": True, "titleFont": "Arial", "labelFont": "Arial"},
        "header": {"titleFontSize": 14, "labelFontSize": 12, "titleFont": "Arial", "labelFont": "Arial"},
        "legend": {"titleFontSize": 14, "labelFontSize": 12, "titleFont": "Arial", "labelFont": "Arial"},
        "range": {"category": PLOTLY_COLORS}
    }
}
alt.themes.register("coffee_theme", lambda: ALTAIR_THEME)
alt.themes.enable("coffee_theme")

# Custom CSS for a clean and professional look
st.markdown("""
<style>
.main {
    background-color: #FDF9F3;
    color: #4A332A;
    font-family: 'Times New Roman', serif;
}
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.2rem;
    font-weight: bold;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 10px 20px;
    background-color: #AB63FA;
    border: 2px solid #636EFA;
    border-bottom: none;
    transition: background-color 0.3s ease;
}
.stTabs [aria-selected="true"] {
    background-color: #EF553B;
    color: white;
    border-color: #EF553B;
}
.metric-container {
    background-color: #00CC96;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #19D3F3;
    text-align: center;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.2);
}
.metric-title {
    font-size: 1.2rem;
    color: #3B221B;
    margin-bottom: 5px;
    font-weight: bold;
}
.metric-value {
    font-size: 2.5rem;
    color: #4A332A;
    font-weight: bold;
}
h1, h2, h3, h4, h5, h6 {
    color: #3B221B;
    font-family: 'Georgia', serif;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --- 2. DATA LOADING AND PREPROCESSING ---
@st.cache_data
def load_data(file_path):
    """Loads the coffee sales data and performs initial cleaning."""
    try:
        df = pd.read_csv(file_path)
        # Convert necessary columns to the correct data type
        df['Date'] = pd.to_datetime(df['Date'])
        # Combine Date and Time for full timestamp with error handling
        df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'], errors='coerce')
        # Drop rows where datetime couldn't be parsed
        df.dropna(subset=['datetime'], inplace=True)
        # Extract time-based features
        df['hour'] = df['datetime'].dt.hour
        df['month'] = df['datetime'].dt.month
        df['month_name'] = df['datetime'].dt.strftime('%B')
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['weekday_name'] = df['datetime'].dt.strftime('%A')
        df['year'] = df['datetime'].dt.year
        
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Fall'
        
        df['season'] = df['month'].apply(get_season)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]) # 5 = Saturday, 6 = Sunday
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure it's in the same directory as the script.")
        return None

def create_dashboard():
    """Builds and displays the entire Streamlit dashboard with tabs."""
    st.title("☕ Professional Coffee Sales Dashboard")
    st.markdown("A comprehensive tool for analyzing key metrics, trends, and seasonal insights from your coffee sales data.")

    df = load_data("Coffe_sales.csv")
    if df is None:
        return

    # --- 3. SIDEBAR FILTERS ---
    st.sidebar.header("Filter Options")
    st.sidebar.markdown("Use the filters below to refine your analysis.")

    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))].copy()
    else:
        st.sidebar.warning("Please select a valid date range.")
        st.stop()

    all_coffees = sorted(df_filtered['coffee_name'].unique())
    selected_coffees = st.sidebar.multiselect("Select Coffee Name(s)", options=all_coffees, default=all_coffees)

    all_cash_types = sorted(df_filtered['cash_type'].unique())
    selected_cash_types = st.sidebar.multiselect("Select Payment Type(s)", options=all_cash_types, default=all_cash_types)
    
    all_seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    selected_seasons = st.sidebar.multiselect("Select Season(s)", options=all_seasons, default=all_seasons)
    
    all_times = sorted(df_filtered['Time_of_Day'].unique())
    selected_times = st.sidebar.multiselect("Select Time of Day", options=all_times, default=all_times)

    df_filtered = df_filtered[df_filtered['coffee_name'].isin(selected_coffees)]
    df_filtered = df_filtered[df_filtered['cash_type'].isin(selected_cash_types)]
    df_filtered = df_filtered[df_filtered['season'].isin(selected_seasons)]
    df_filtered = df_filtered[df_filtered['Time_of_Day'].isin(selected_times)]
    
    if df_filtered.empty:
        st.warning("No data available for the selected filters. Please adjust your selections.")
        return

    # --- 4. TABS & CONTENT ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["KPIs", "Monthly Trends", "Seasonal Analysis", "Product Analysis", "Advanced Insights", "Model Analysis"])

    # --- TAB 1: KPIs ---
    with tab1:
        st.header("Key Performance Indicators")
        st.markdown("A high-level overview of total sales, number of orders, and average transaction value.")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_sales = df_filtered['money'].sum()
            st.markdown(f'<div class="metric-container"><p class="metric-title">Total Sales</p><p class="metric-value">${total_sales:,.2f}</p></div>', unsafe_allow_html=True)

        with col2:
            total_transactions = df_filtered.shape[0]
            st.markdown(f'<div class="metric-container"><p class="metric-title">Total Orders</p><p class="metric-value">{total_transactions:,.0f}</p></div>', unsafe_allow_html=True)
        
        with col3:
            avg_transaction_value = df_filtered['money'].mean()
            st.markdown(f'<div class="metric-container"><p class="metric-title">Avg. Order Value</p><p class="metric-value">${avg_transaction_value:,.2f}</p></div>', unsafe_allow_html=True)
            
        st.markdown("---")
        st.header("Weekday vs. Weekend Performance")
        st.markdown("Compare key metrics between weekdays and weekends to identify different customer behaviors.")

        weekday_weekend_sales = df_filtered.groupby('is_weekend')['money'].sum().reset_index()
        weekday_weekend_sales['is_weekend'] = weekday_weekend_sales['is_weekend'].map({True: 'Weekend', False: 'Weekday'})
        
        weekday_weekend_counts = df_filtered.groupby('is_weekend').size().reset_index(name='Total Orders')
        weekday_weekend_counts['is_weekend'] = weekday_weekend_counts['is_weekend'].map({True: 'Weekend', False: 'Weekday'})
        
        col_t1_4, col_t1_5, col_t1_6 = st.columns(3)
        with col_t1_4:
            # Chart 1: Total Sales by Weekday vs. Weekend (Bar Chart)
            base = alt.Chart(weekday_weekend_sales).encode(
                x=alt.X('is_weekend', title='Day Type'),
                y=alt.Y('money', title='Total Sales ($)'),
                tooltip=[alt.Tooltip('is_weekend', title='Day Type'), alt.Tooltip('money', format='$,.2f', title='Total Sales')]
            ).properties(title="Total Sales: Weekday vs. Weekend")
            chart1 = base.mark_bar(color=PLOTLY_COLORS[0])
            text = base.mark_text(
                align='center',
                baseline='bottom',
                dy=-5, # Nudge the text
                color='black'
            ).encode(text=alt.Text('money', format='$,.2f'))
            st.altair_chart(chart1 + text, use_container_width=True)

        with col_t1_5:
            # Chart 2: Total Orders by Weekday vs. Weekend (Bar Chart)
            base = alt.Chart(weekday_weekend_counts).encode(
                x=alt.X('is_weekend', title='Day Type'),
                y=alt.Y('Total Orders', title='Total Orders'),
                tooltip=[alt.Tooltip('is_weekend', title='Day Type'), 'Total Orders']
            ).properties(title="Total Orders: Weekday vs. Weekend")
            chart2 = base.mark_bar(color=PLOTLY_COLORS[1])
            text = base.mark_text(
                align='center',
                baseline='bottom',
                dy=-5,
                color='black'
            ).encode(text=alt.Text('Total Orders', format=',.0f'))
            st.altair_chart(chart2 + text, use_container_width=True)


    # --- TAB 2: MONTHLY TRENDS ---
    with tab2:
        st.header("Monthly and Daily Trends")
        st.markdown("Track sales and orders over time to identify growth patterns and seasonal shifts.")

        col_t2_1, col_t2_2 = st.columns(2)
        with col_t2_1:
            # Chart 4: Monthly Revenue Trend (Line Chart)
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
            monthly_sales = df_filtered.groupby('month_name')['money'].sum().reindex(month_order).reset_index()
            base = alt.Chart(monthly_sales).encode(
                x=alt.X('month_name', sort=month_order, title='Month'),
                y=alt.Y('money', title='Total Revenue ($)'),
                tooltip=['month_name', alt.Tooltip('money', format='$,.2f')]
            ).properties(title="Monthly Revenue Trend")
            line_chart = base.mark_line(point=True, color=PLOTLY_COLORS[0])
            text = base.mark_text(
                align='left',
                baseline='middle',
                dx=5,
                color='black'
            ).encode(text=alt.Text('money', format='$,.0f'))
            st.altair_chart(line_chart + text, use_container_width=True)

        with col_t2_2:
            # Chart 5: Monthly Orders Trend (Bar Chart)
            monthly_orders = df_filtered.groupby('month_name').size().reindex(month_order).reset_index(name='count')
            base = alt.Chart(monthly_orders).encode(
                x=alt.X('month_name', sort=month_order, title='Month'),
                y=alt.Y('count', title='Total Orders'),
                tooltip=['month_name', 'count']
            ).properties(title="Monthly Order Trend")
            chart5 = base.mark_bar(color=PLOTLY_COLORS[1])
            text = base.mark_text(
                align='center',
                baseline='bottom',
                dy=-5,
                color='black'
            ).encode(text=alt.Text('count', format=',.0f'))
            st.altair_chart(chart5 + text, use_container_width=True)

        st.markdown("---")
        st.header("Daily and Cumulative Analysis")
        
        col_t2_3, col_t2_4 = st.columns(2)
        with col_t2_3:
            # Chart 6: Cumulative Revenue Over Time (Area Chart)
            df_filtered_sorted = df_filtered.sort_values('datetime')
            df_filtered_sorted['cumulative_sales'] = df_filtered_sorted['money'].cumsum()
            chart6 = alt.Chart(df_filtered_sorted).mark_area(color=PLOTLY_COLORS[2], opacity=0.7).encode(
                x=alt.X('datetime', title='Date'),
                y=alt.Y('cumulative_sales', title='Cumulative Revenue ($)'),
                tooltip=[alt.Tooltip('datetime', title='Date', format="%Y-%m-%d"), alt.Tooltip('cumulative_sales', title='Cumulative Revenue', format='$,.2f')]
            ).properties(title="Cumulative Revenue Trend")
            st.altair_chart(chart6, use_container_width=True)
       

    # --- TAB 3: SEASONAL ANALYSIS ---
    with tab3:
        st.header("Seasonal Sales and Orders")
        st.markdown("Detailed breakdown of revenue and orders by season to highlight seasonal trends.")
        
        seasonal_summary = df_filtered.groupby('season').agg(
            total_sales=('money', 'sum'),
            total_orders=('money', 'size')
        ).reset_index()
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.subheader("Winter")
            winter_sales = seasonal_summary[seasonal_summary['season'] == 'Winter']['total_sales'].sum()
            st.metric("Total Sales", f"${winter_sales:,.2f}")
        with col_s2:
            st.subheader("Spring")
            spring_sales = seasonal_summary[seasonal_summary['season'] == 'Spring']['total_sales'].sum()
            st.metric("Total Sales", f"${spring_sales:,.2f}")
        with col_s3:
            st.subheader("Summer")
            summer_sales = seasonal_summary[seasonal_summary['season'] == 'Summer']['total_sales'].sum()
            st.metric("Total Sales", f"${summer_sales:,.2f}")
        with col_s4:
            st.subheader("Fall")
            fall_sales = seasonal_summary[seasonal_summary['season'] == 'Fall']['total_sales'].sum()
            st.metric("Total Sales", f"${fall_sales:,.2f}")

        st.markdown("---")

        col_s5, col_s6 = st.columns(2)
        with col_s5:
            # Chart 9: Total Revenue by Season (Vertical Bar Chart)
            base = alt.Chart(seasonal_summary).encode(
                x=alt.X('season', sort=['Winter', 'Spring', 'Summer', 'Fall'], title='Season'),
                y=alt.Y('total_sales', title='Total Revenue ($)'),
                tooltip=['season', alt.Tooltip('total_sales', format='$,.2f')]
            ).properties(title="Total Revenue by Season")
            chart9 = base.mark_bar(color=PLOTLY_COLORS[2])
            text = base.mark_text(
                align='center',
                baseline='bottom',
                dy=-5,
                color='black'
            ).encode(text=alt.Text('total_sales', format='$,.2f'))
            st.altair_chart(chart9 + text, use_container_width=True)
        with col_s6:
            # Chart 10: Total Orders by Season (Vertical Bar Chart)
            base = alt.Chart(seasonal_summary).encode(
                x=alt.X('season', sort=['Winter', 'Spring', 'Summer', 'Fall'], title='Season'),
                y=alt.Y('total_orders', title='Total Orders'),
                tooltip=['season', alt.Tooltip('total_orders', format=',.0f')]
            ).properties(title="Total Orders by Season")
            chart10 = base.mark_bar(color=PLOTLY_COLORS[3])
            text = base.mark_text(
                align='center',
                baseline='bottom',
                dy=-5,
                color='black'
            ).encode(text=alt.Text('total_orders', format=',.0f'))
            st.altair_chart(chart10 + text, use_container_width=True)

        col_s7, col_s8 = st.columns(2)
        with col_s7:
            # Chart 11: Seasonal Revenue Distribution (Donut Chart)
            chart11 = alt.Chart(seasonal_summary).mark_arc(outerRadius=120, innerRadius=80, stroke="#fff").encode(
                theta=alt.Theta("total_sales", stack=True),
                color=alt.Color("season", title="Season", scale=alt.Scale(range=PLOTLY_COLORS)),
                tooltip=["season", alt.Tooltip("total_sales", title="Total Revenue", format='$,.2f')]
            )
            text = chart11.mark_text(radius=140).encode(
                text=alt.Text("total_sales", format='$,.0f'),
                order=alt.Order("total_sales", sort="descending"),
                color=alt.value('black')
            )
            st.altair_chart(chart11 + text, use_container_width=True)
        with col_s8:
            # Chart 12: Seasonal Orders Distribution (Donut Chart)
            chart12 = alt.Chart(seasonal_summary).mark_arc(outerRadius=120, innerRadius=80, stroke="#fff").encode(
                theta=alt.Theta("total_orders", stack=True),
                color=alt.Color("season", title="Season", scale=alt.Scale(range=PLOTLY_COLORS)),
                tooltip=["season", alt.Tooltip("total_orders", title="Total Orders")]
            )
            text = chart12.mark_text(radius=140).encode(
                text=alt.Text("total_orders", format=',.0f'),
                order=alt.Order("total_orders", sort="descending"),
                color=alt.value('black')
            )
            st.altair_chart(chart12 + text, use_container_width=True)

        st.markdown("---")
        st.header("Seasonal Product Insights")
        st.markdown("See which products are most popular in each season.")
        
        product_seasonal_sales = df_filtered.groupby(['season', 'coffee_name'])['money'].sum().reset_index()
        
        # Chart 13: Top 5 Bestselling Coffees by Season (Grouped Bar Chart)
        top_5_per_season = product_seasonal_sales.loc[product_seasonal_sales.groupby('season')['money'].nlargest(5).index.get_level_values(1)]
        chart13 = alt.Chart(top_5_per_season).mark_bar().encode(
            x=alt.X('money', title='Total Revenue ($)'),
            y=alt.Y('coffee_name', sort='-x', title='Coffee Name'),
            color=alt.Color('season', scale=alt.Scale(range=PLOTLY_COLORS)),
            column=alt.Column('season', header=alt.Header(titleOrient="bottom", labelOrient="bottom")),
            tooltip=['season', 'coffee_name', alt.Tooltip('money', format='$,.2f', title='Total Revenue')]
        ).properties(title="Top 5 Coffees by Revenue per Season")
        st.altair_chart(chart13, use_container_width=True)

        # Chart 14: Percentage of Revenue by Coffee Name per Season (100% Stacked Bar Chart)
        chart14 = alt.Chart(product_seasonal_sales).mark_bar().encode(
            x=alt.X('money', stack="normalize", title='Percentage of Revenue (%)', axis=alt.Axis(format='%')),
            y=alt.Y('season', sort=['Winter', 'Spring', 'Summer', 'Fall'], title='Season'),
            color=alt.Color('coffee_name', title='Coffee Name', scale=alt.Scale(range=PLOTLY_COLORS)),
            tooltip=['season', 'coffee_name', alt.Tooltip('money', format='$,.2f', title='Total Revenue')]
        ).properties(title="Revenue Contribution of Each Coffee by Season")
        st.altair_chart(chart14, use_container_width=True)

    # --- TAB 4: PRODUCT ANALYSIS ---
    with tab4:
        st.header("Product Performance & Payment Methods")
        st.markdown("Analysis of top-selling products, average prices, and preferred payment methods.")
        
        col_t4_1, col_t4_2 = st.columns(2)
        with col_t4_1:
            # Chart 15: Top 10 Bestselling Coffees by Revenue (Horizontal Bar Chart)
            top_coffee_sales = df_filtered.groupby('coffee_name')['money'].sum().nlargest(10).reset_index()
            top_coffee_sales.columns = ['Coffee Name', 'Total Sales']
            base = alt.Chart(top_coffee_sales).encode(
                x=alt.X('Total Sales', title='Total Revenue ($)'),
                y=alt.Y('Coffee Name', sort='-x', title=''),
                tooltip=['Coffee Name', alt.Tooltip('Total Sales', format='$,.2f')]
            ).properties(title="Top 10 Bestselling Coffees")
            chart15 = base.mark_bar(color=PLOTLY_COLORS[2])
            text = base.mark_text(
                align='left',
                dx=3,
                color='black'
            ).encode(text=alt.Text('Total Sales', format='$,.2f'))
            st.altair_chart(chart15 + text, use_container_width=True)
        with col_t4_2:
            # Chart 16: Top 10 Most Ordered Coffees (Horizontal Bar Chart)
            top_coffee_count = df_filtered['coffee_name'].value_counts().nlargest(10).reset_index()
            top_coffee_count.columns = ['Coffee Name', 'Transaction Count']
            base = alt.Chart(top_coffee_count).encode(
                x=alt.X('Transaction Count', title='Number of Orders'),
                y=alt.Y('Coffee Name', sort='-x', title=''),
                tooltip=['Coffee Name', 'Transaction Count']
            ).properties(title="Top 10 Most Ordered Coffees")
            chart16 = base.mark_bar(color=PLOTLY_COLORS[3])
            text = base.mark_text(
                align='left',
                dx=3,
                color='black'
            ).encode(text=alt.Text('Transaction Count', format=',.0f'))
            st.altair_chart(chart16 + text, use_container_width=True)

        st.markdown("---")
        
        col_t4_3, col_t4_4 = st.columns(2)
        with col_t4_3:
            # Chart 17: Revenue by Payment Type (Pie Chart)
            sales_by_payment = df_filtered.groupby('cash_type')['money'].sum().reset_index()
            chart17 = alt.Chart(sales_by_payment).mark_arc(outerRadius=120, stroke="#fff").encode(
                theta=alt.Theta("money", stack=True),
                color=alt.Color("cash_type", title="Payment Type", scale=alt.Scale(range=PLOTLY_COLORS)),
                tooltip=["cash_type", alt.Tooltip("money", title="Total Revenue", format='$,.2f')]
            )
            text = chart17.mark_text(radius=140).encode(
                text=alt.Text("money", format='$,.0f'),
                order=alt.Order("money", sort="descending"),
                color=alt.value('black')
            )
            st.altair_chart(chart17 + text, use_container_width=True)
        with col_t4_4:
            # Chart 18: Average Price per Coffee (Horizontal Bar Chart)
            avg_price_coffee = df_filtered.groupby('coffee_name')['money'].mean().sort_values(ascending=False).reset_index()
            base = alt.Chart(avg_price_coffee).encode(
                x=alt.X('money', title='Average Price ($)'),
                y=alt.Y('coffee_name', sort='-x', title=''),
                tooltip=['coffee_name', alt.Tooltip('money', format='$,.2f')]
            ).properties(title="Average Price per Coffee")
            chart18 = base.mark_bar(color=PLOTLY_COLORS[4])
            text = base.mark_text(
                align='left',
                dx=3,
                color='black'
            ).encode(text=alt.Text('money', format='$,.2f'))
            st.altair_chart(chart18 + text, use_container_width=True)

        st.markdown("---")
        
        # Chart 19: Revenue by Product and Time of Day (Stacked Bar Chart)
        revenue_by_product_time = df_filtered.groupby(['coffee_name', 'Time_of_Day'])['money'].sum().reset_index()
        chart19 = alt.Chart(revenue_by_product_time).mark_bar().encode(
            x=alt.X('money', title='Total Revenue ($)'),
            y=alt.Y('coffee_name', sort='-x', title=''),
            color=alt.Color('Time_of_Day', scale=alt.Scale(range=PLOTLY_COLORS)),
            tooltip=['coffee_name', 'Time_of_Day', alt.Tooltip('money', format='$,.2f', title='Total Revenue')]
        ).properties(title="Revenue by Product and Time of Day")
        st.altair_chart(chart19, use_container_width=True)

    # --- TAB 5: ADVANCED INSIGHTS ---
    with tab5:
        st.header("Time-Based Insights")
        st.markdown("Analyze revenue and order counts by various time periods.")

        col_t5_1, col_t5_2 = st.columns(2)
        with col_t5_1:
            # Chart 20: Revenue by Day of Week (Vertical Bar Chart)
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            sales_by_day_of_week = df_filtered.groupby('weekday_name')['money'].sum().reindex(weekday_order).reset_index()
            base = alt.Chart(sales_by_day_of_week).encode(
                x=alt.X('weekday_name', sort=weekday_order, title='Day of Week'),
                y=alt.Y('money', title='Total Revenue ($)'),
                tooltip=['weekday_name', alt.Tooltip('money', format='$,.2f')]
            ).properties(title="Total Revenue by Day of Week")
            chart20 = base.mark_bar(color=PLOTLY_COLORS[2])
            text = base.mark_text(
                align='center',
                baseline='bottom',
                dy=-5,
                color='black'
            ).encode(text=alt.Text('money', format='$,.2f'))
            st.altair_chart(chart20 + text, use_container_width=True)
        with col_t5_2:
            # Chart 21: Orders by Day of Week (Vertical Bar Chart)
            orders_by_day_of_week = df_filtered.groupby('weekday_name').size().reindex(weekday_order).reset_index(name='count')
            base = alt.Chart(orders_by_day_of_week).encode(
                x=alt.X('weekday_name', sort=weekday_order, title='Day of Week'),
                y=alt.Y('count', title='Total Orders'),
                tooltip=['weekday_name', 'count']
            ).properties(title="Total Orders by Day of Week")
            chart21 = base.mark_bar(color=PLOTLY_COLORS[3])
            text = base.mark_text(
                align='center',
                baseline='bottom',
                dy=-5,
                color='black'
            ).encode(text=alt.Text('count', format=',.0f'))
            st.altair_chart(chart21 + text, use_container_width=True)

        st.markdown("---")

        col_t5_3, col_t5_4 = st.columns(2)
        with col_t5_3:
            # Chart 22: Revenue by Hour of Day (Vertical Bar Chart)
            sales_by_hour = df_filtered.groupby('hour')['money'].sum().reset_index()
            base = alt.Chart(sales_by_hour).encode(
                x=alt.X('hour', title='Hour of Day'),
                y=alt.Y('money', title='Total Revenue ($)'),
                tooltip=['hour', alt.Tooltip('money', format='$,.2f')]
            ).properties(title="Total Revenue by Hour of Day")
            chart22 = base.mark_bar(color=PLOTLY_COLORS[2])
            text = base.mark_text(
                align='center',
                baseline='bottom',
                dy=-5,
                color='black'
            ).encode(text=alt.Text('money', format='$,.2f'))
            st.altair_chart(chart22 + text, use_container_width=True)
        with col_t5_4:
            # Chart 23: Orders by Hour of Day (Vertical Bar Chart)
            orders_by_hour = df_filtered.groupby('hour').size().reset_index(name='count')
            base = alt.Chart(orders_by_hour).encode(
                x=alt.X('hour', title='Hour of Day'),
                y=alt.Y('count', title='Total Orders'),
                tooltip=['hour', 'count']
            ).properties(title="Total Orders by Hour of Day")
            chart23 = base.mark_bar(color=PLOTLY_COLORS[3])
            text = base.mark_text(
                align='center',
                baseline='bottom',
                dy=-5,
                color='black'
            ).encode(text=alt.Text('count', format=',.0f'))
            st.altair_chart(chart23 + text, use_container_width=True)

        st.markdown("---")
        # Chart 24: Revenue Heatmap by Weekday and Hour
        revenue_heatmap_data = df_filtered.groupby(['weekday_name', 'hour'])['money'].sum().reset_index()
        chart24 = alt.Chart(revenue_heatmap_data).mark_rect().encode(
            x=alt.X('hour', title='Hour of Day'),
            y=alt.Y('weekday_name', title='Day of Week'),
            color=alt.Color('money', title='Total Revenue ($)', scale=alt.Scale(scheme='viridis')),
            tooltip=['weekday_name', 'hour', alt.Tooltip('money', format='$,.2f', title='Total Revenue')]
        ).properties(title="Revenue Heatmap: Day of Week vs. Hour")
        st.altair_chart(chart24, use_container_width=True)
        
        st.markdown("---")
        
        col_t5_5, col_t5_6 = st.columns(2)
        with col_t5_5:
            # Chart 25: Distribution of Transaction Values (Histogram)
            base = alt.Chart(df_filtered).encode(
                alt.X("money", bin=True, title='Transaction Value ($)'),
                alt.Y('count()', title='Number of Transactions'),
                tooltip=['count()']
            ).properties(title="Distribution of Transaction Values")
            chart25 = base.mark_bar(color=PLOTLY_COLORS[1])
            text = base.mark_text(
                align='center',
                baseline='bottom',
                dy=-5,
                color='black'
            ).encode(text=alt.Text('count()', format=',.0f'))
            st.altair_chart(chart25 + text, use_container_width=True)

        st.markdown("---")
        
        col_t5_7, col_t5_8 = st.columns(2)
        with col_t5_7:
            # Chart 27: Correlation Matrix
            numeric_df = df_filtered[['money', 'hour', 'day_of_week']].corr().stack().reset_index()
            numeric_df.columns = ['Variable 1', 'Variable 2', 'Correlation']
            chart27 = alt.Chart(numeric_df).mark_rect().encode(
                x=alt.X('Variable 1', title=''),
                y=alt.Y('Variable 2', title=''),
                color=alt.Color('Correlation', scale=alt.Scale(range=[PLOTLY_COLORS[0], 'white', PLOTLY_COLORS[2]]), title='Correlation'),
                tooltip=['Variable 1', 'Variable 2', alt.Tooltip('Correlation', format='.2f')]
            ).properties(title="Correlation Matrix")
            st.altair_chart(chart27, use_container_width=True)

    # --- TAB 6: MODEL ANALYSIS ---
    with tab6:
        
        # --- Data Prep for Model ---
        df_model = df.copy()
        
        # Date Features (to match the user's provided code)
        df_model['Date'] = pd.to_datetime(df_model['Date'], format='%Y-%m-%d')
        df_model['Year'] = df_model['Date'].dt.year.astype(str)
        df_model['Month'] = df_model['Date'].dt.month
        df_model['Season'] = df_model['Month'].map({
            12:'Winter',1:'Winter',2:'Winter',
            3:'Spring',4:'Spring',5:'Spring',
            6:'Summer',7:'Summer',8:'Summer',
            9:'Autumn',10:'Autumn',11:'Autumn'
        })
        
        # Target variable (log transform to reduce skewness)
        y = np.log1p(df_model["money"])

        # Features
        categorical_cols = ['coffee_name', 'Time_of_Day', 'Year', 'Season']
        numerical_cols = ["hour", "day_of_week", "month"]

        # Handle potential missing columns after filtering
        categorical_cols = [col for col in categorical_cols if col in df_model.columns]
        numerical_cols = [col for col in numerical_cols if col in df_model.columns]

        # Encoding
        X_cat = pd.get_dummies(df_model[categorical_cols], drop_first=True, dtype=int)
        
        # Standardize numerical features
        if not numerical_cols:
            X_num = pd.DataFrame(index=df_model.index)
        else:
            X_num = (df_model[numerical_cols] - df_model[numerical_cols].mean()) / df_model[numerical_cols].std()
        
        X_combined = pd.concat([X_num, X_cat], axis=1)

        # Drop any columns that may have all zeros
        X_combined = X_combined.loc[:, (X_combined != 0).any(axis=0)]
        
        # Add a constant (intercept) to the model
        X_final = sm.add_constant(X_combined)

        # --- Train/Test Split ---
        n = len(X_final)
        train_size = int(0.8 * n)
        
        np.random.seed(42)  # For reproducibility
        idx = np.random.permutation(n)
        train_idx, test_idx = idx[:train_size], idx[train_size:]

        x_train, x_test = X_final.iloc[train_idx], X_final.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        if x_train.empty or y_train.empty:
            st.warning("Not enough data to train the model with the current filters. Please broaden your date range.")
        else:
            # --- Model Training ---
           
            model = sm.OLS(y_train, x_train).fit()

            # Create input widgets for user
            col_input1, col_input2, col_input3 = st.columns(3)
            with col_input1:
                input_coffee = st.selectbox("Coffee Name", options=sorted(df_model['coffee_name'].unique()))
                input_cash_type = st.selectbox("Payment Type", options=sorted(df_model['cash_type'].unique()))
            with col_input2:
                input_time_of_day = st.selectbox("Time of Day", options=sorted(df_model['Time_of_Day'].unique()))
                input_year = st.selectbox("Year", options=sorted(df_model['year'].unique()))
            with col_input3:
                input_month = st.selectbox("Month", options=sorted(df_model['month'].unique()))
                input_day_of_week = st.selectbox("Day of Week", options=sorted(df_model['day_of_week'].unique()))
                input_hour = st.slider("Hour of Day", 0, 23, 12)

            # Button to trigger prediction
            if st.button("Predict Sales"):
                # Prepare user input for prediction
                input_df = pd.DataFrame(columns=x_train.columns, index=[0])
                input_df.loc[0] = 0 # Initialize with zeros
                input_df.loc[0, 'const'] = 1
                
                # Handle categorical inputs with one-hot encoding
                if f'coffee_name_{input_coffee}' in input_df.columns:
                    input_df[f'coffee_name_{input_coffee}'] = 1
                if f'Time_of_Day_{input_time_of_day}' in input_df.columns:
                    input_df[f'Time_of_Day_{input_time_of_day}'] = 1
                if f'cash_type_{input_cash_type}' in input_df.columns:
                    input_df[f'cash_type_{input_cash_type}'] = 1

                season = 'Winter' if input_month in [12, 1, 2] else 'Spring' if input_month in [3, 4, 5] else 'Summer' if input_month in [6, 7, 8] else 'Fall'
                if f'Season_{season}' in input_df.columns:
                    input_df[f'Season_{season}'] = 1
                if f'Year_{input_year}' in input_df.columns:
                    input_df[f'Year_{input_year}'] = 1

                # Handle numerical inputs
                numerical_features = ["hour", "month", "day_of_week"]
                numerical_data = {
                    "hour": input_hour,
                    "month": input_month,
                    "day_of_week": input_day_of_week,
                }

                # Standardize numerical inputs using the training set's mean and std
                for col in numerical_features:
                    if col in x_train.columns:
                        mean_val = df_model[col].mean()
                        std_val = df_model[col].std()
                        standardized_value = (numerical_data[col] - mean_val) / std_val
                        input_df.loc[0, col] = standardized_value

                # Ensure all columns are in the correct order as the training data
                input_df = input_df[x_train.columns]
                
                # Make prediction
                log_prediction = model.predict(input_df)[0]
                prediction = np.expm1(log_prediction) # Inverse transform log prediction
                
                st.subheader(f"Predicted Sales: ${prediction:.2f}")
            

            st.subheader("Model Performance on Test Set")
            y_pred = model.predict(x_test)

            # Manually calculate R2 and RMSE
            ssr = np.sum((y_test - y_pred)**2)
            sst = np.sum((y_test - np.mean(y_test))**2)
            r2 = 1 - (ssr / sst)
            rmse = np.sqrt(np.mean((y_test - y_pred)**2))

            st.markdown(f"**R² Score**: `{r2:.4f}`")
            st.markdown(f"**Root Mean Squared Error (RMSE)**: `{rmse:.4f}`")


if __name__ == "__main__":
    create_dashboard()
