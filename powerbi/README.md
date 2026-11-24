# Power BI dashboard

Import `data/processed/clean_sales.csv` as **FactSales**. Create a calendar table and connect `Calendar[Date]` to `FactSales[order_date]`. Import `test_forecasts.csv` for forecast-quality visuals.

Suggested pages:

1. **Executive overview**: Revenue, Profit, Profit Margin, Orders, Average Order Value and Revenue Growth KPIs with monthly trend.
2. **Product & location**: Category contribution, top products, state/store proxy comparison and discount versus profit scatterplot.
3. **Forecast quality**: Actual versus predicted sales, MAPE by category and MAPE by state/store proxy.

The source dataset does not include physical stores. The pipeline derives the `store` analytical proxy from `state`, and labels it consistently in the dataset outputs.
