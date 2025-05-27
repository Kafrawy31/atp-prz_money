import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import plotly.express as px
import plotly.graph_objects as go

@st.cache_data
def load_data():
    df = pd.read_csv("earnings_data.csv", parse_dates=["rankdate"])
    
    df['Rank Bin (5s)'] = pd.cut(
        df['sglrank'],
        bins=range(1, 506, 5),  # Start at 1, up to and including 500
        right=False,            # Left-inclusive: [1–5)
        labels=[f"{i}-{i+4}" for i in range(1, 501, 5)],
        include_lowest=True
    )

    # 10-rank bins
    df['Rank Bin (10s)'] = pd.cut(
        df['sglrank'],
        bins=range(1, 511, 10),  # Start at 1
        right=False,
        labels=[f"{i}-{i+9}" for i in range(1, 501, 10)],
        include_lowest=True
    )

    # extract year
    df['Year'] = df['rankdate'].dt.year

    # ensure bins are categorical
    df['Rank Bin (5s)']  = df['Rank Bin (5s)'].astype('category')
    df['Rank Bin (10s)'] = df['Rank Bin (10s)'].astype('category')

    # helper to sort bin labels by their numeric lower bound
    def sorted_bin_categories(series: pd.Series) -> list[str]:
        cats = series.cat.categories
        # parse "A-B" → A as int, then sort
        sorted_cats = sorted(cats, key=lambda lbl: int(lbl.split('-')[0]))
        return sorted_cats

    # apply ordering back to the series
    df['Rank Bin (5s)']  = df['Rank Bin (5s)'].cat.set_categories(
        sorted_bin_categories(df['Rank Bin (5s)']),
        ordered=True
    )
    df['Rank Bin (10s)'] = df['Rank Bin (10s)'].cat.set_categories(
        sorted_bin_categories(df['Rank Bin (10s)']),
        ordered=True
    )

    return df

df = load_data()

# --- Sidebar filters ---
st.sidebar.title("Filters")

# Year filter
years = sorted(df['Year'].unique())
selected_years = st.sidebar.multiselect(
    "Select Year(s)",
    options=years,
    default=years
)

# Bin type filter
bin_type = st.sidebar.radio("Rank Bin Type", ["5s", "10s"])
sgl_rank_min = int(df['sglrank'].min())
sgl_rank_max = int(df['sglrank'].max())

use_rank_filter = st.sidebar.checkbox("Use Rank Range Filter")
use_adjusted_earnings = st.sidebar.checkbox("Show expected (adjusted) prize money for 2025", value=False)

if use_adjusted_earnings:
    filtered = df[df['Year'] == 2024].copy()
    earnings_column = 'Net Prize Money (2025 Adjusted)'
else:
    earnings_column = 'Net Prize Money (Actual)'

if use_rank_filter:
    sgl_rank_range = st.sidebar.slider(
        "Select SGL Rank Range",
        min_value=sgl_rank_min,
        max_value=sgl_rank_max,
        value=(sgl_rank_min, sgl_rank_max)
    )

if use_rank_filter:
    # Filter by rank range only
    filtered = df[
        (df['Year'].isin(selected_years)) &
        (df['sglrank'] >= sgl_rank_range[0]) &
        (df['sglrank'] <= sgl_rank_range[1])
    ]
else:
    # Filter by bin type
    filtered = df[df['Year'].isin(selected_years)]
    if bin_type == "5s":
        selected_bin = st.sidebar.selectbox(
            "Select Rank Bin (5s)",
            options=filtered['Rank Bin (5s)'].cat.categories
        )
        filtered = filtered[filtered['Rank Bin (5s)'] == selected_bin]
    else:
        selected_bin = st.sidebar.selectbox(
            "Select Rank Bin (10s)",
            options=filtered['Rank Bin (10s)'].cat.categories
        )
        filtered = filtered[filtered['Rank Bin (10s)'] == selected_bin]

earnings = filtered[earnings_column].sort_values().reset_index(drop=True)

# --- Main title ---
st.title("ATP Player Earnings by Rank Bin")
if use_rank_filter:
    st.markdown(
        f"**Years:** {', '.join(map(str, selected_years))} • "
        f"**SGL Rank Range:** {sgl_rank_range[0]} – {sgl_rank_range[1]}"
    )
else:
    st.markdown(
        f"**Years:** {', '.join(map(str, selected_years))} • "
        f"**Rank Bin:** {selected_bin} ({bin_type})"
    )


if earnings.empty:
    st.warning("No data for selected combination.")
else:
    # 1) Histogram of counts
    st.subheader("Histogram of Net Prize Money (Counts)")
    fig_hist = px.histogram(
        filtered,
        x='Net Prize Money (Actual)',
        nbins=8,
        title="Prize Money Distribution (Counts)",
        labels={"Net Prize Money (Actual)": "Prize Money"},
    )
    fig_hist.update_layout(
        xaxis_tickformat=',',
        xaxis_tickangle=90,
        yaxis_title="Count"
    )
    st.plotly_chart(fig_hist)

    # 2) Density curve only
    st.subheader("Density Curve of Net Prize Money")
    earnings_m = earnings / 1e6   # convert dollars → millions of dollars
    mean_val = earnings_m.mean()
    std_val = earnings_m.std()

    lower_bound = mean_val - std_val
    upper_bound = mean_val + std_val
    
    kde = gaussian_kde(earnings_m)
    x_range = np.linspace(earnings_m.min(), earnings_m.max(), 200)
    y_kde = kde(x_range)
    fig_kde = px.line(
        x=x_range,
        y=y_kde,
        labels={'x': 'Net Prize Money (Actual)', 'y': 'Density'},
        title="Prize Money Density Curve with ±1 SD"
    )

    fig_kde.add_vline(x=mean_val, line_color="green", line_dash="dot",
                    annotation_text=f"Mean: ${mean_val:,.3f} m")
    fig_kde.add_vline(x=lower_bound, line_color="red", line_dash="dash",
                    annotation_text=f"−1 SD: ${lower_bound:,.3f}m")
    fig_kde.add_vline(x=upper_bound, line_color="red", line_dash="dash",
                    annotation_text=f"+1 SD: ${upper_bound:,.3f}m")

    fig_kde.update_layout(xaxis_tickformat=',', xaxis_tickangle=90)
    st.plotly_chart(fig_kde)
        
    mean_display = f"""
    <div style="
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    ">
        Mean: {mean_val * 1e6:,.2f}
    </div>
    """

    upper_display = f"""
    <div style="
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    ">
        +1 SD: {upper_bound * 1e6:,.2f}
    </div>
    """

    lower_display = f"""
    <div style="
        background-color: #cce5ff;
        color: #004085;
        border: 1px solid #b8daff;
        border-radius: 0.25rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    ">
        −1 SD: {lower_bound * 1e6:,.2f}
    </div>
    """

    combined_display = f"""
    <div style='display: flex; gap: 1rem;'>
    {mean_display}
    {upper_display}
    {lower_display}
    </div>
    """

    st.markdown(combined_display, unsafe_allow_html=True)
    
    # Calculate the actual % of data within ±1 SD
    within_1sd = earnings_m[(earnings_m >= lower_bound) & (earnings_m <= upper_bound)]
    percent_within_1sd = (len(within_1sd) / len(earnings_m)) * 100

    # Display the % as a styled info box
    st.info(f"Percentage of players within ±1 SD: {percent_within_1sd:.2f}%")


    # 3) ECDF & median
    st.subheader("Empirical Cumulative Distribution Function (ECDF)")
    
    ecdf_y = (earnings.rank(method='first') / len(earnings)).values
    ecdf_x = earnings.values
    median_val = earnings.median()
    ecdf_title = (
    f"ECDF – Rank Bin {selected_bin}"
    if not use_rank_filter
    else f"ECDF – SGL Rank Range {sgl_rank_range[0]} – {sgl_rank_range[1]}"
)
    fig_ecdf = px.line(
        x=ecdf_x,
        y=ecdf_y,
        labels={'x': 'Net Prize Money (Actual)', 'y': 'Cumulative Proportion'},
        title=ecdf_title
    )
    fig_ecdf.add_vline(
        x=median_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Median: {median_val:,.0f}"
    )
    fig_ecdf.update_layout(xaxis_tickformat=',', xaxis_tickangle=90)
    st.plotly_chart(fig_ecdf)

    st.success(f"Median Net Prize Money: {median_val:,.0f}")
