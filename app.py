import numpy as np
import pandas as pd 
import plotly.express as px
import streamlit as st
from datetime import datetime


st.set_page_config(page_title="Interactive Dashboard (Dummy Data)", layout='wide')


@st.cache_data
def make_dummy_data(n_rows: int =50000, seed: int =42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    dates = pd.date_range(end= pd.Timestamp.today().normalize(), periods=180, freq="D")
    # dates = pd.date_range(
    # start="2023-01-01",
    # end="2023-06-30",
    # freq="D")
    date = rng.choice(dates, size= n_rows)

    regions = np.array(["NY/NJ", "West", "South", "Midwest"])
    products = np.array(["Equities", "Fixed Income", "Options", "Futures"])
    desks = np.array(["Desk A", "Desk B", "Desk C"])
    channels = np.array(["API", "UI", "Batch"])

    df =pd.DataFrame(

    {
    "date": pd.to_datetime(date),
    "region": rng.choice(regions, size= n_rows, p= [0.35, 0.25, 0.20, 0.20]),
    "product": rng.choice(products, size= n_rows),
    "desk": rng.choice(desks, size= n_rows),
    "channel": rng.choice(channels, size= n_rows, p= [0.45, 0.35, 0.20]),
    "volume": rng.integers(1, 200 , size= n_rows),
    "revenue": np.round(rng.normal(120, 40, size= n_rows).clip(5, None), 2),
    "cost": np.round(rng.normal(70, 25, size= n_rows).clip(2, None), 2),
    }

    )

    df["pnl"] = np.round(df["revenue"] - df["cost"], 2)

    base = rng.normal(0.02, 0.01, size = n_rows).clip(0, 0.08)
    product_bump = df["product"].map({"Equities": 0.01, "Fixed Income": 0.005, "options": 0.015, "Futures": 0.012}).values
    df["alert_rate"] = np.round((base + product_bump).clip(0, 0.08), 4)

    return df 


st.title("Interactive dashboard (Dummy data)")
st.caption("Streamlit + Plotly example with filters , KPIs, trends, breakdown, and a detail table")

df= make_dummy_data()

min_d, max_d = df["date"].min().date(), df["date"].max().date()

with st.sidebar:
    st.header("Filters")

    date_range = st.date_input("Date range", value = [min_d, max_d], min_value= min_d, max_value=max_d)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range[0] , date_range[1]

    else: 
        start_date = date_range
        end_date = date_range
        # min_d, max_d
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    region_sel= st.multiselect("Region", sorted(df["region"].unique().tolist()), default=sorted(df["region"].unique().tolist()))
    product_sel= st.multiselect("Product", sorted(df["product"].unique().tolist()), default= sorted(df["product"].unique().tolist()))
    desk_sel= st.multiselect("Desk", sorted(df["desk"].unique().tolist()), default=sorted(df["desk"].unique().tolist()))
    channel_sel= st.multiselect("Channel", sorted(df["channel"].unique().tolist()), default=sorted(df["channel"].unique().tolist()))

    if not region_sel: region_sel = df["region"].unique().tolist()
    if not product_sel: product_sel = df["product"].unique().tolist()
    if not desk_sel: product_sel = df["desk"].unique().tolist()
    if not channel_sel: channel_sel = df["channel"].unique().tolist()
    
        
        
    st.divider()
    metric = st.selectbox("Primary metric", ["pnl", "revenue", "cost", "volume", "alert_rate"])
    agg = st.selectbox("Aggregation", ["sum", "mean"])
    show_top_n = st.slider("Top N categories", 5, 20, 10)

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)  # make end inclusive

mask = (
    (df["date"] >= start_ts)
    & (df["date"] < end_ts)
    & (df["region"].isin(region_sel))
    & (df["product"].isin(product_sel))
    & (df["desk"].isin(desk_sel))
    & (df["channel"].isin(channel_sel))
)
f = df.loc[mask].copy()

if f.empty:
    st.warning("No date for the selected filters.Try widening the date range or selecting more categories.")
    st.stop()
    


kpi1, kpi2, kpi3, kpi4 = st.columns(4)

total_volume = int(f["volume"].sum())
total_revenue = float(f["revenue"].sum())
total_cost= float(f["cost"].sum())
total_pnl = float(f["pnl"].sum())
avg_alert = float(f["alert_rate"].mean())

kpi1.metric("Total Volume", f"{total_volume:,}")
kpi2.metric("Total Revenue", f"${total_revenue:,.0f}")
kpi3.metric("Total pnl", f"${total_pnl:,.0f}")
kpi4.metric("Avg Alert Rate", f"{avg_alert*100:.2f}%")

st.divider()

daily = (
    f.groupby("date", as_index=False)
    .agg(
        pnl= ("pnl", "sum"),
        cost =("cost", "sum"),
        volume = ("volume", "sum"),
        alert_rate= ("alert_rate", "mean"),
    )
    .sort_values("date")

)

fig_trend = px.line(
    daily,
    x= "date",
    y= metric,
    title =f"Daily Trend: {metric} ({agg})",
    markers= False,
)

if agg == "mean":

    daily[f"{metric}_roll7"] = daily[metric].rolling(7, min_periods=1).mean()
    fig_trend = px.line(
        daily, 
        x = "date",
        y= f"{metric}_roll7",
        title = f"Daily Trend: {metric} (7-day rolling mean)",
    )

c1, c2 = st.columns(2)

def top_breakdown(df_in: pd.DataFrame, group_col: str, metric_col: str, agg_fn: str, n: int) -> pd.DataFrame:
    if agg_fn == "sum":
        out = df_in.groupby(group_col, as_index=False)[metric_col].sum()
    else:
        out = df_in.groupby(group_col, as_index=False)[metric_col].mean()
    return out.sort_values(metric_col, ascending=False).head(n)

by_product = top_breakdown(f, "product", metric, agg, show_top_n)
by_region = top_breakdown(f, "region", metric, agg, show_top_n)

with c1:
    fig_prod = px.bar(by_product, x= "product", y =metric , title= f"{metric} by Product ({agg}, top {show_top_n}")
    st.plotly_chart(fig_prod , use_container_width=True)

with c2:
    fig_reg = px.bar(by_region, x ="region", y=metric , title=f"{metric} by region ({agg}, top {show_top_n})")
    st.plotly_chart(fig_reg, use_container_width =True)

st.subheader("PnL vs Volume (sample)")
sample = f.sample(n=min(len(f) , 3000), random_state =7)
fig_scatter = px.scatter(
    sample,
    x= "volume",
    y= "pnl",
    color = "product",
    hover_data = ["region", "desk", "channel", "date"],
    title = "Scatter (Sampled for speed): volume vs pnl",
)

st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("Detail Table")
st.dataframe(
    f.sort_values("date", ascending=False).head(2000),
    use_container_width=True,
)

csv = f.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered data (csv)", data= csv, file_name= "filtered_dummy_data.csv", mime="text/csv")      
                               
    