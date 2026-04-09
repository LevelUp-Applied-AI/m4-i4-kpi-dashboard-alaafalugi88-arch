import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sqlalchemy import create_engine


# =========================
# CONNECT
# =========================
def connect_db():
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@localhost:5432/amman_market"
    )
    return create_engine(db_url)


# =========================
# EXTRACT DATA
# =========================
def extract_data(engine):
    customers = pd.read_sql("SELECT * FROM customers", engine)
    products = pd.read_sql("SELECT * FROM products", engine)
    orders = pd.read_sql("SELECT * FROM orders", engine)
    order_items = pd.read_sql("SELECT * FROM order_items", engine)

    return {
        "customers": customers,
        "products": products,
        "orders": orders,
        "order_items": order_items
    }


# =========================
# COMPUTE KPIs
# =========================
def compute_kpis(data_dict):
    customers = data_dict["customers"]
    products = data_dict["products"]
    orders = data_dict["orders"]
    order_items = data_dict["order_items"]

    # تنظيف
    orders = orders[orders["status"] != "cancelled"]
    order_items = order_items[order_items["quantity"] <= 100]

    # join
    df = orders.merge(order_items, on="order_id") \
               .merge(products, on="product_id") \
               .merge(customers, on="customer_id")

    df["revenue"] = df["quantity"] * df["unit_price"]
    df["order_date"] = pd.to_datetime(df["order_date"])

    # KPI 1: Monthly Revenue
    monthly_revenue = df.groupby(df["order_date"].dt.to_period("M"))["revenue"].sum()

    # KPI 2: Revenue Growth
    revenue_growth = monthly_revenue.pct_change()

    # KPI 3: Average Order Value
    order_total = df.groupby("order_id")["revenue"].sum()
    aov = order_total.mean()

    # KPI 4: Revenue by City
    revenue_city = df.groupby("city")["revenue"].sum()

    # KPI 5: Revenue by Category
    revenue_category = df.groupby("category")["revenue"].sum()

    return {
        "monthly_revenue": monthly_revenue,
        "revenue_growth": revenue_growth,
        "aov": aov,
        "revenue_city": revenue_city,
        "revenue_category": revenue_category,
        "full_df": df
    }


# =========================
# STATISTICAL TESTS
# =========================
def run_statistical_tests(data_dict):
    kpis = compute_kpis(data_dict)
    df = kpis["full_df"]

    # t-test between cities
    amman = df[df["city"] == "Amman"]["revenue"]
    irbid = df[df["city"] == "Irbid"]["revenue"]

    t_stat, p_val = stats.ttest_ind(amman, irbid, equal_var=False)

    return {
        "t_test_city": {
            "t_stat": t_stat,
            "p_value": p_val,
            "interpretation": "Significant difference" if p_val < 0.05 else "No significant difference"
        }
    }


# =========================
# VISUALIZATIONS
# =========================
def create_visualizations(kpi_results, stat_results):
    sns.set_palette("colorblind")

    # 1. Monthly Revenue
    plt.figure()
    kpi_results["monthly_revenue"].plot()
    plt.title("Monthly Revenue Trend")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.savefig("output/monthly_revenue.png")
    plt.close()

    # 2. Revenue Growth
    plt.figure()
    kpi_results["revenue_growth"].plot()
    plt.title("Revenue Growth Rate")
    plt.savefig("output/revenue_growth.png")
    plt.close()

    # 3. Revenue by City
    plt.figure()
    kpi_results["revenue_city"].plot(kind="bar")
    plt.title("Revenue by City")
    plt.savefig("output/revenue_by_city.png")
    plt.close()

    # 4. Revenue by Category
    plt.figure()
    kpi_results["revenue_category"].plot(kind="bar")
    plt.title("Revenue by Category")
    plt.savefig("output/revenue_by_category.png")
    plt.close()

    # 5. Boxplot
    plt.figure()
    sns.boxplot(x="category", y="revenue", data=kpi_results["full_df"])
    plt.title("Order Value Distribution by Category")
    plt.savefig("output/boxplot.png")
    plt.close()


# =========================
# MAIN
# =========================
def main():
    os.makedirs("output", exist_ok=True)

    engine = connect_db()
    data = extract_data(engine)

    kpis = compute_kpis(data)
    stats_results = run_statistical_tests(data)

    create_visualizations(kpis, stats_results)

    print("AOV:", kpis["aov"])
    print("Stat Test:", stats_results)


if __name__ == "__main__":
    main()
