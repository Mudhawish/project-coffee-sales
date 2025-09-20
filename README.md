# ☕ Coffee Sales Analysis & Prediction

An interactive data analysis and prediction project using **coffee shop transaction data** (March 2024 → March 2025).  
The project includes **EDA, statistical analysis, regression models, and a Streamlit dashboard**.

---

## 📂 Project Structure
```text
├── Coffe_sales.csv                        # Dataset
├── Project_coffe_sales_with_models_new.ipynb  # Jupyter Notebook (EDA + Modeling)
├── dashboard.py                                 # Streamlit app for dashboard
├── requirements.txt                       # Dependencies
└── README.md                              # Project documentation
```




---

## 🎯 Objectives
- **Analyze** sales patterns by time, weekday, month, and season.  
- **Identify** top-selling coffee types and customer spending habits.  
- **Compare** weekday vs weekend performance.  
- **Build** predictive models (OLS, Linear Regression, XGBoost).  
- **Deploy** results via a Streamlit dashboard.  

---

## 📊 Dataset
- **Source**: Kaggle Coffee Shop Transactions Dataset  
- **Period Covered**: March 2024 → March 2025 (1 full year)  
- **Transactions**: 3,500  

| Column        | Description |
|---------------|-------------|
| `hour_of_day` | Hour of purchase (0–23) |
| `cash_type`   | Payment type (Cash/Card) |
| `money`       | Transaction amount |
| `coffee_name` | Coffee type (Latte, Americano, etc.) |
| `Time_of_Day` | Morning / Afternoon / Night |
| `Weekday`     | Day of the week |
| `Month_name`  | Month name |
| `Weekdaysort` | Numeric weekday (1=Mon … 7=Sun) |
| `Monthsort`   | Numeric month (1=Jan … 12=Dec) |
| `Date`        | Transaction date |
| `Time`        | Transaction time |

📌 [Dataset Link](https://www.kaggle.com/datasets/navjotkaushal/coffee-sales-dataset)

---

## 🛠️ Installation

Clone the repository:
```bash
git clone https://github.com/MUSAB10000/Project-Coffee-Sales.git
cd Project-Coffee-Sales
```

------
📈 Key Insights

Sales are evenly split between weekdays and weekends (~50/50).

Autumn recorded the highest seasonal sales (~29%).

Americano with Milk and Latte are the most popular coffee types.

XGBoost provided the best predictive accuracy compared to OLS and Linear Regression.
