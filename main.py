import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import plotly.express as px

st.title("ATP Player Earnings by Rank Bin")

# --- File uploader ---
uploaded_file = st.file_uploader("Please upload the earnings data CSV file:", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["rankdate"])

    # Create bins for 5s and 10s
    df['Rank Bin (5s)'] = pd.cut(
        df['sglrank'],
        bins=range(1, 506, 5),
        right=False,
        labels=[f"{i}-{i+4}" for i in range(1, 501, 5)],
        include_lowest=True
    )
    df['Rank Bin (10s)'] = pd.cut(
        df['sglrank'],
        bins=range(1, 511, 10),
        right=False,
        labels=[f"{i}-{i+9}" for i in range(1, 501, 10)],
        include_lowest=True
    )
    df['Year'] = df['rankdate'].dt.year

    # Set bins as ordered categorical
    def sorted_bin_categories(series: pd.Series) -> list[str]:
        cats = series.cat.categories
        return sorted(cats, key=lambda lbl: int(lbl.split('-')[0]))

    df['Rank Bin (5s)'] = df['Rank Bin (5s)'].astype('category').cat.set_categories(
        sorted_bin_categories(df['Rank Bin (5s)']), ordered=True)
    df['Rank Bin (10s)'] = df['Rank Bin (10s)'].astype('category').cat.set_categories(
        sorted_bin_categories(df['Rank Bin (10s)']), ordered=True)

    # --- Sidebar filters ---
    st.sidebar.title("Filters")
    use_adjusted_earnings = st.sidebar.checkbox("Show expected (adjusted) prize money for following year", value=False)

    # Determine year column based on earnings type
    year_column = 'Baseline Year' if use_adjusted_earnings else 'Year'
    years = sorted(df[year_column].unique())
    selected_years = st.sidebar.multiselect("Select Year(s)", options=years, default=years)

    earnings_column = 'Net Prize Money (2025 Adjusted)' if use_adjusted_earnings else 'Net Prize Money (Actual)'

    bin_type = st.sidebar.radio("Rank Bin Type", ["5s", "10s"])

    sgl_rank_min, sgl_rank_max = int(df['sglrank'].min()), int(df['sglrank'].max())
    use_rank_filter = st.sidebar.checkbox("Use Rank Range Filter")
    if use_rank_filter:
        st.sidebar.write("Enter SGL Rank Range:")
        sgl_rank_min_input = st.sidebar.number_input("Min SGL Rank", value=sgl_rank_min, min_value=sgl_rank_min, max_value=sgl_rank_max)
        sgl_rank_max_input = st.sidebar.number_input("Max SGL Rank", value=sgl_rank_max, min_value=sgl_rank_min, max_value=sgl_rank_max)
        sgl_rank_range = (sgl_rank_min_input, sgl_rank_max_input)

    use_snumtrn_filter = st.sidebar.checkbox("Use Tourament Filter")
    if use_snumtrn_filter:
        snumtrn_min, snumtrn_max = int(df['snumtrn'].min()), int(df['snumtrn'].max())
        snumtrn_range = st.sidebar.slider("Select number of touraments Range", min_value=snumtrn_min, max_value=snumtrn_max, value=(snumtrn_min, snumtrn_max))

    use_carprz_filter = st.sidebar.checkbox("Use career prize Filter")
    if use_carprz_filter:
        carprz_min, carprz_max = int(df['carprz'].min()), int(df['carprz'].max())
        st.sidebar.write("Enter career prize Range:")
        carprz_min_input = st.sidebar.number_input("Min career prize", value=carprz_min, min_value=carprz_min, max_value=carprz_max)
        carprz_max_input = st.sidebar.number_input("Max career prize", value=carprz_max, min_value=carprz_min, max_value=carprz_max)
        carprz_range = (carprz_min_input, carprz_max_input)

    use_prize_money_filter = st.sidebar.checkbox("Use Prize Money Filter")
    if use_prize_money_filter:
        prize_min, prize_max = int(df[earnings_column].min()), int(df[earnings_column].max())
        st.sidebar.write("Enter Prize Money Range:")
        prize_min_input = st.sidebar.number_input("Min Prize Money", value=prize_min, min_value=prize_min, max_value=prize_max)
        prize_max_input = st.sidebar.number_input("Max Prize Money", value=prize_max, min_value=prize_min, max_value=prize_max)
        prize_range = (prize_min_input, prize_max_input)

    # --- Filtering logic ---
    filtered = df[df[year_column].isin(selected_years)].copy()
    selected_bin = None

    if use_rank_filter:
        filtered = filtered[(filtered['sglrank'] >= sgl_rank_range[0]) & (filtered['sglrank'] <= sgl_rank_range[1])]
    else:
        selected_bin = st.sidebar.selectbox(
            f"Select Rank Bin ({bin_type})",
            options=filtered[f'Rank Bin ({bin_type})'].cat.categories
        )
        filtered = filtered[filtered[f'Rank Bin ({bin_type})'] == selected_bin]

    if use_snumtrn_filter:
        filtered = filtered[(filtered['snumtrn'] >= snumtrn_range[0]) & (filtered['snumtrn'] <= snumtrn_range[1])]

    if use_carprz_filter:
        filtered = filtered[(filtered['carprz'] >= carprz_range[0]) & (filtered['carprz'] <= carprz_range[1])]

    if use_prize_money_filter:
        filtered = filtered[(filtered[earnings_column] >= prize_range[0]) & (filtered[earnings_column] <= prize_range[1])]

    earnings = filtered[earnings_column].sort_values().reset_index(drop=True)

    # --- Rest of the script remains unchanged ---
    # (Main title, histogram, density plot, ECDF, etc.)


    # --- Main title ---
    if use_adjusted_earnings:
        st.markdown("**Year:** 2024 • **Expected (Adjusted) Earnings**")
    elif use_rank_filter:
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
        # 1) Histogram
        st.subheader("Histogram of Net Prize Money (Counts)")
        fig_hist = px.histogram(
            filtered,
            x=earnings_column,
            nbins=15,
            title="Prize Money Distribution (Counts)",
            labels={earnings_column: "Prize Money"},
        )
        fig_hist.update_layout(xaxis_tickformat=',', xaxis_tickangle=90, yaxis_title="Count")
        st.plotly_chart(fig_hist)

        # 2) Density curve
        st.subheader("Density Curve of Net Prize Money")
        earnings_m = earnings / 1e6
        mean_val = earnings_m.mean()
        std_val = earnings_m.std()
        lower_bound = max(mean_val - std_val, 0)
        upper_bound = mean_val + std_val

        kde = gaussian_kde(earnings_m)
        x_range = np.linspace(earnings_m.min(), earnings_m.max(), 200)
        y_kde = kde(x_range)
        fig_kde = px.line(
            x=x_range,
            y=y_kde,
            labels={'x': 'Net Prize Money (Millions)', 'y': 'Density'},
            title="Prize Money Density Curve with ±1 SD"
        )
        fig_kde.add_vline(x=mean_val, line_color="green", line_dash="dot",
                          annotation_text=f"Mean: ${mean_val:,.3f} m")
        fig_kde.add_vline(x=lower_bound, line_color="red", line_dash="dash",
                          annotation_text=f"−1 SD: ${lower_bound:,.3f}m")
        fig_kde.add_vline(x=upper_bound, line_color="red", line_dash="dash",
                          annotation_text=f"+1 SD: ${upper_bound:,.3f}m")
        fig_kde.update_layout(xaxis_tickformat=',', xaxis_tickangle=90)
        fig_kde.update_traces(
    hovertemplate='Net Prize Money (Millions): %{x:.3f}<br>Density: %{y:.3f}<extra></extra>'
)
    
        st.plotly_chart(fig_kde)


        # Mean, SD displays
        mean_display = f"""<div style="background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb;
        border-radius: 0.25rem; padding: 0.5rem 1rem; font-weight: bold;">Mean: ${mean_val * 1e6:,.0f}</div>"""
        upper_display = f"""<div style="background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;
        border-radius: 0.25rem; padding: 0.5rem 1rem; font-weight: bold;">+1 SD: ${upper_bound * 1e6:,.0f}</div>"""
        lower_display = f"""<div style="background-color: #cce5ff; color: #004085; border: 1px solid #b8daff;
        border-radius: 0.25rem; padding: 0.5rem 1rem; font-weight: bold;">−1 SD: ${lower_bound * 1e6:,.0f}</div>"""
        combined_display = f"""<div style='display: flex; gap: 1rem;'>{mean_display}{upper_display}{lower_display}</div>"""
        st.markdown(combined_display, unsafe_allow_html=True)

        # % within ±1 SD
        within_1sd = earnings_m[(earnings_m >= lower_bound) & (earnings_m <= upper_bound)]
        percent_within_1sd = (len(within_1sd) / len(earnings_m)) * 100
        st.info(f"Percentage of players within ±1 SD: {percent_within_1sd:.2f}%")

        # 3) ECDF
        st.subheader("Empirical Cumulative Distribution Function (ECDF)")
        ecdf_y = (earnings.rank(method='first') / len(earnings)).values
        ecdf_x = earnings.values
        median_val = earnings.median()
        if not use_rank_filter:
            ecdf_title = f"ECDF – Rank Bin {selected_bin}"
        elif use_rank_filter:
            ecdf_title = f"ECDF – SGL Rank Range {sgl_rank_range[0]} – {sgl_rank_range[1]}"
        else:
            ecdf_title = "ECDF – Expected Prize Money (2025)"

        fig_ecdf = px.line(
            x=ecdf_x,
            y=ecdf_y,
            labels={'x': earnings_column, 'y': 'Cumulative Proportion'},
            title=ecdf_title
        )
        fig_ecdf.add_vline(x=median_val, line_dash="dash", line_color="red",
                           annotation_text=f"Median: {median_val:,.0f}")
        fig_ecdf.update_layout(xaxis_tickformat=',', xaxis_tickangle=90)
        st.plotly_chart(fig_ecdf)
        st.success(f"Median Net Prize Money: {median_val:,.0f}")
else:
    st.warning("Please upload the earnings data file to proceed.")
