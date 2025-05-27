import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import plotly.express as px
import plotly.graph_objects as go

st.title("ATP Player Earnings Analysis")

# --- File uploader ---
uploaded_file = st.file_uploader("Please upload the earnings data CSV file:", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["rankdate"])
    df['Year'] = df['rankdate'].dt.year
    df['Baseline Year'] = df['rankdate'].dt.year + 1

    # Create bins for 5s and 10s
    # Ensure sglrank is numeric and handle potential NaNs before max()
    sglrank_numeric = pd.to_numeric(df['sglrank'], errors='coerce')
    if not sglrank_numeric.isna().all() and sglrank_numeric.max() >=1 :
        max_rank = int(sglrank_numeric.max())
        df['Rank Bin (5s)'] = pd.cut(
            df['sglrank'],
            bins=range(1, max_rank + 6, 5),
            right=False,
            labels=[f"{i}-{i+4}" for i in range(1, max_rank + 1, 5)],
            include_lowest=True
        )
        df['Rank Bin (10s)'] = pd.cut(
            df['sglrank'],
            bins=range(1, max_rank + 11, 10),
            right=False,
            labels=[f"{i}-{i+9}" for i in range(1, max_rank + 1, 10)],
            include_lowest=True
        )
    
        def sorted_bin_categories(series: pd.Series) -> list[str]:
            cats = series.dropna().astype(str).unique()
            cats = [c for c in cats if c != 'nan']
            return sorted(cats, key=lambda lbl: int(lbl.split('-')[0]))

        if 'Rank Bin (5s)' in df.columns and not df['Rank Bin (5s)'].isna().all():
            df['Rank Bin (5s)'] = df['Rank Bin (5s)'].astype('category').cat.set_categories(
                sorted_bin_categories(df['Rank Bin (5s)']), ordered=True)
        if 'Rank Bin (10s)' in df.columns and not df['Rank Bin (10s)'].isna().all():
            df['Rank Bin (10s)'] = df['Rank Bin (10s)'].astype('category').cat.set_categories(
                sorted_bin_categories(df['Rank Bin (10s)']), ordered=True)
    else:
        st.warning("Single ranks ('sglrank') column has issues or no valid data; binning might be affected.")


    # --- Sidebar filters ---
    st.sidebar.title("Filters")
    use_adjusted_earnings = st.sidebar.checkbox("Show expected (adjusted) prize money for following year", value=False)

    year_column = 'Baseline Year' if use_adjusted_earnings else 'Year'
    
    if year_column in df.columns and df[year_column].nunique() > 0:
        years = sorted(df[year_column].unique())
        default_years = years
    else:
        years = []
        default_years = []
        st.sidebar.warning(f"No data available for '{year_column}'. Please check the column in your CSV.")

    selected_years = st.sidebar.multiselect(f"Select {year_column}(s)", options=years, default=default_years)

    earnings_column = 'Net Prize Money (2025 Adjusted)' if use_adjusted_earnings else 'Net Prize Money (Actual)'
    
    # Essential columns check for core functionality
    essential_cols = ['Net Prize Money (2025 Adjusted)', 'Net Prize Money (Actual)', 'sglrank']
    missing_essential_cols = [col for col in essential_cols if col not in df.columns]
    if missing_essential_cols:
        st.error(f"Essential column(s) not found: {', '.join(missing_essential_cols)}. Please check the uploaded file.")
        st.stop()
    
    # Columns needed for mandatory shortfall conditions
    shortfall_condition_cols = ['snumtrn', 'carprz']
    for col in shortfall_condition_cols:
        if col not in df.columns:
            st.warning(f"Column '{col}' not found. Shortfall calculations requiring this column will treat its condition as not met by any player.")


    # --- Expected Shortfall Summary (Baseline Year 2025) ---
    st.subheader("Expected Shortfall Summary (for Baseline Year 2025, using Adjusted Earnings)")
    st.markdown("_Players must have >14 tournaments played and < £15M career prize money to be included in shortfall._")
    df_baseline_2025 = df[df['Baseline Year'] == 2025].copy() # Use .copy()
    
    count1_expected = 0
    shortfall1_expected = 0
    count2_expected = 0
    shortfall2_expected = 0

    if not df_baseline_2025.empty:
        base_condition1_expected = (
            df_baseline_2025['sglrank'].between(101, 175)
            & (df_baseline_2025['Net Prize Money (2025 Adjusted)'] < 200_000)
        )
        base_condition2_expected = (
            df_baseline_2025['sglrank'].between(176, 250)
            & (df_baseline_2025['Net Prize Money (2025 Adjusted)'] < 100_000)
        )

        # snumtrn condition for expected shortfall
        if 'snumtrn' in df_baseline_2025.columns:
            snumtrn_filter_expected = (df_baseline_2025['snumtrn'] > 14)
        else:
            # Warning already issued globally if column missing from df
            snumtrn_filter_expected = pd.Series(False, index=df_baseline_2025.index)
        
        # carprz condition for expected shortfall
        if 'carprz' in df_baseline_2025.columns:
            carprz_filter_expected = (df_baseline_2025['carprz'] < 15_000_000)
        else:
            # Warning already issued globally if column missing from df
            carprz_filter_expected = pd.Series(False, index=df_baseline_2025.index)

        mask1_expected = base_condition1_expected & snumtrn_filter_expected & carprz_filter_expected
        count1_expected = mask1_expected.sum()
        shortfall1_expected = (200_000 - df_baseline_2025.loc[mask1_expected, 'Net Prize Money (2025 Adjusted)']).sum()

        mask2_expected = base_condition2_expected & snumtrn_filter_expected & carprz_filter_expected
        count2_expected = mask2_expected.sum()
        shortfall2_expected = (100_000 - df_baseline_2025.loc[mask2_expected, 'Net Prize Money (2025 Adjusted)']).sum()
    
    c1_exp, c2_exp = st.columns(2)
    c1_exp.metric("Ranks 101–175: # below £200k", count1_expected, f"£{shortfall1_expected:,.0f} total expected shortfall")
    c2_exp.metric("Ranks 176–250: # below £100k", count2_expected, f"£{shortfall2_expected:,.0f} total expected shortfall")
    st.markdown("---")

    bin_type = st.sidebar.radio("Rank Bin Type", ["5s", "10s"])
    rank_bin_col_name = f'Rank Bin ({bin_type})'

    sgl_rank_min_df, sgl_rank_max_df = int(sglrank_numeric.dropna().min()), int(sglrank_numeric.dropna().max())
    use_rank_filter = st.sidebar.checkbox("Use Rank Range Filter")
    sgl_rank_range = (sgl_rank_min_df, sgl_rank_max_df)
    if use_rank_filter:
        st.sidebar.write("Enter SGL Rank Range:")
        sgl_rank_min_input = st.sidebar.number_input("Min SGL Rank", value=sgl_rank_min_df, min_value=sgl_rank_min_df, max_value=sgl_rank_max_df)
        sgl_rank_max_input = st.sidebar.number_input("Max SGL Rank", value=sgl_rank_max_df, min_value=sgl_rank_min_df, max_value=sgl_rank_max_df)
        sgl_rank_range = (sgl_rank_min_input, sgl_rank_max_input)

    use_snumtrn_filter = st.sidebar.checkbox("Use Tournament Filter (snumtrn)")
    snumtrn_range = None
    if use_snumtrn_filter and 'snumtrn' in df.columns and not df['snumtrn'].dropna().empty:
        snumtrn_min, snumtrn_max = int(df['snumtrn'].dropna().min()), int(df['snumtrn'].dropna().max())
        snumtrn_range = st.sidebar.slider("Select number of tournaments Range", min_value=snumtrn_min, max_value=snumtrn_max, value=(snumtrn_min, snumtrn_max))
    elif use_snumtrn_filter:
        st.sidebar.warning("'snumtrn' column not found or empty; cannot apply tournament filter.")

    use_carprz_filter = st.sidebar.checkbox("Use career prize Filter (carprz)")
    carprz_range = None
    if use_carprz_filter and 'carprz' in df.columns and not df['carprz'].dropna().empty:
        carprz_min, carprz_max = int(df['carprz'].dropna().min()), int(df['carprz'].dropna().max())
        st.sidebar.write("Enter career prize Range:")
        carprz_min_input = st.sidebar.number_input("Min career prize", value=carprz_min, min_value=carprz_min, max_value=carprz_max)
        carprz_max_input = st.sidebar.number_input("Max career prize", value=carprz_max, min_value=carprz_min, max_value=carprz_max)
        carprz_range = (carprz_min_input, carprz_max_input)
    elif use_carprz_filter:
        st.sidebar.warning("'carprz' column not found or empty; cannot apply career prize filter.")

    use_prize_money_filter = st.sidebar.checkbox("Use Prize Money Filter")
    prize_range = None
    if use_prize_money_filter and earnings_column in df.columns:
        if not df[earnings_column].dropna().empty:
            prize_min, prize_max = int(df[earnings_column].dropna().min()), int(df[earnings_column].dropna().max())
            st.sidebar.write("Enter Prize Money Range:")
            prize_min_input = st.sidebar.number_input("Min Prize Money", value=prize_min, min_value=prize_min, max_value=prize_max)
            prize_max_input = st.sidebar.number_input("Max Prize Money", value=prize_max, min_value=prize_min, max_value=prize_max)
            prize_range = (prize_min_input, prize_max_input)
        else:
            st.sidebar.warning(f"No data in '{earnings_column}' for prize money filter.")
            use_prize_money_filter = False
    elif use_prize_money_filter:
        st.sidebar.warning(f"'{earnings_column}' column not found for prize money filter.")

    # --- Filtering logic ---
    filtered = df.copy()
    if year_column in df.columns and selected_years:
         filtered = df[df[year_column].isin(selected_years)].copy()
    elif not selected_years and year_column in df.columns and df[year_column].nunique() > 0:
        st.info(f"No specific {year_column}(s) selected. Displaying data for all available {year_column}s.")
        # 'filtered' remains a copy of full df, or year-filtered if selection was cleared then re-populated
    elif not selected_years:
         st.warning(f"No {year_column}(s) selected or available. Data shown might be incomplete or for all years.")


    selected_bin = None
    if use_rank_filter:
        if 'sglrank' in filtered.columns:
            filtered = filtered[(filtered['sglrank'] >= sgl_rank_range[0]) & (filtered['sglrank'] <= sgl_rank_range[1])]
    else:
        if rank_bin_col_name in filtered.columns and filtered[rank_bin_col_name].notna().any() and hasattr(filtered[rank_bin_col_name],'cat') :
            bin_options = filtered[rank_bin_col_name].cat.categories
            if not bin_options.empty:
                selected_bin = st.sidebar.selectbox(
                    f"Select Rank Bin ({bin_type})",
                    options=bin_options
                )
                if selected_bin:
                    filtered = filtered[filtered[rank_bin_col_name] == selected_bin]
            else:
                st.sidebar.warning(f"No rank bins available for {bin_type} with current filters.")
        elif rank_bin_col_name in df.columns : # Column exists in df but maybe not in filtered or no categories
             st.sidebar.warning(f"Rank bin column '{rank_bin_col_name}' has no categories for current selection.")
        # else: # Column for binning doesn't exist at all
            # Initial warning about binning problems should cover this.

    if use_snumtrn_filter and snumtrn_range and 'snumtrn' in filtered.columns:
        filtered = filtered[(filtered['snumtrn'] >= snumtrn_range[0]) & (filtered['snumtrn'] <= snumtrn_range[1])]
    if use_carprz_filter and carprz_range and 'carprz' in filtered.columns:
        filtered = filtered[(filtered['carprz'] >= carprz_range[0]) & (filtered['carprz'] <= carprz_range[1])]
    if use_prize_money_filter and prize_range and earnings_column in filtered.columns:
        filtered = filtered[(filtered[earnings_column] >= prize_range[0]) & (filtered[earnings_column] <= prize_range[1])]
    
    earnings = pd.Series(dtype=float)
    if earnings_column in filtered.columns and not filtered[earnings_column].dropna().empty:
        earnings = filtered[earnings_column].dropna().sort_values().reset_index(drop=True)

    # --- Main title ---
    title_year_prefix = "Baseline Year(s):" if use_adjusted_earnings else "Year(s):"
    title_earnings_suffix = "Expected (Adjusted) Earnings" if use_adjusted_earnings else "Actual Earnings"
    
    years_display_string_main = "None Selected"
    if selected_years:
        all_available_years_for_mode = df[year_column].dropna().unique()
        if len(all_available_years_for_mode) > 0 and len(selected_years) == len(all_available_years_for_mode):
            years_display_string_main = f"All ({year_column})"
        else:
            years_display_string_main = ', '.join(map(str, sorted(selected_years)))
    elif year_column in df.columns and df[year_column].nunique() > 0 :
         years_display_string_main = f"All ({year_column} - Default)"

    main_title_text = ""
    if use_rank_filter:
        main_title_text = (
            f"**{title_year_prefix}** {years_display_string_main} • "
            f"**SGL Rank Range:** {sgl_rank_range[0]} – {sgl_rank_range[1]} • **{title_earnings_suffix}**"
        )
    else: 
        main_title_text = (
            f"**{title_year_prefix}** {years_display_string_main} • "
            f"**Rank Bin:** {selected_bin if selected_bin else 'N/A'} ({bin_type}) • **{title_earnings_suffix}**"
        )
    st.markdown(main_title_text)

    if earnings.empty:
        st.warning("No data available for the selected filter combination.")
    else:
        # --- Plot Titles Setup ---
        plot_title_status = "Adjusted" if use_adjusted_earnings else "Actual"
        years_str_for_plot_title = ""
        if selected_years:
            all_years_in_original_df_for_current_mode = df[year_column].dropna().unique()
            if len(all_years_in_original_df_for_current_mode) > 0 and \
               len(selected_years) == len(all_years_in_original_df_for_current_mode) :
                years_str_for_plot_title = f"All {year_column}s"
            else:
                years_str_for_plot_title = f"{year_column}(s): {', '.join(map(str, sorted(selected_years)))}"
        elif year_column in df.columns and df[year_column].nunique() > 0: 
            years_str_for_plot_title = f"All Available {year_column}s"
        else:
            years_str_for_plot_title = f"No {year_column}s Data"

        st.subheader("Prize Money Distribution")
        counts, bin_edges = np.histogram(earnings, bins=15)
        max_count = max(counts) if len(counts) > 0 else 0
        padded_max = max_count * 1.8 if max_count > 0 else 10

        fig_hist = px.histogram(
            x=earnings, nbins=15, labels={'x': earnings_column, 'y': 'Count'}, text_auto=True
        )
        fig_hist.update_layout(
            title_text=f"Distribution of {plot_title_status} Prize Money for {years_str_for_plot_title}",
            yaxis=dict(range=[0, padded_max]), yaxis_title="Count", xaxis_tickformat=',', xaxis_tickangle=90
        )
        fig_hist.update_traces(hovertemplate=f'{earnings_column}: %{{x}}<br>Count: %{{y}}<extra></extra>')
        st.plotly_chart(fig_hist)

        st.subheader(f"Density Curve of {plot_title_status} Net Prize Money")
        earnings_m = earnings / 1e6
        mean_val = earnings_m.median()
        std_val = earnings_m.std()
        lower_bound = max(mean_val - std_val, 0) if not pd.isna(mean_val) and not pd.isna(std_val) else 0
        upper_bound = mean_val + std_val if not pd.isna(mean_val) and not pd.isna(std_val) else 0
        
        if not earnings_m.empty and earnings_m.nunique() > 1:
            kde = gaussian_kde(earnings_m)
            x_range_kde = np.linspace(earnings_m.min(), earnings_m.max(), 200)
            y_kde = kde(x_range_kde)
            fig_kde = px.line(x=x_range_kde, y=y_kde, labels={'x': f'{earnings_column} (Millions)', 'y': 'Density'})
            fig_kde.update_layout(
                 title_text=f"Density of {plot_title_status} Prize Money for {years_str_for_plot_title} (with ±1 SD)",
                 xaxis_tickformat=',.3f', xaxis_tickangle=90
            )
            fig_kde.add_vline(x=mean_val, line_color="green", line_dash="dot", annotation_text=f"Mean: ${mean_val:,.3f}m")
            fig_kde.add_vline(x=lower_bound, line_color="red", line_dash="dash", annotation_text=f"−1 SD: ${lower_bound:,.3f}m")
            fig_kde.add_vline(x=upper_bound, line_color="red", line_dash="dash", annotation_text=f"+1 SD: ${upper_bound:,.3f}m")
            fig_kde.update_traces(hovertemplate=f'{earnings_column} (Millions): %{{x:.3f}}<br>Density: %{{y:.3f}}<extra></extra>')
            st.plotly_chart(fig_kde)
        else:
            st.info("Not enough data points or variance to generate a density curve.")

        mean_display = f"""<div style="background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb;
        border-radius: 0.25rem; padding: 0.5rem 1rem; font-weight: bold;">Median: ${mean_val * 1e6:,.0f}</div>"""
        upper_display = f"""<div style="background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;
        border-radius: 0.25rem; padding: 0.5rem 1rem; font-weight: bold;">+1 SD: ${upper_bound * 1e6:,.0f}</div>"""
        lower_display = f"""<div style="background-color: #cce5ff; color: #004085; border: 1px solid #b8daff;
        border-radius: 0.25rem; padding: 0.5rem 1rem; font-weight: bold;">−1 SD: ${lower_bound * 1e6:,.0f}</div>"""
        combined_display = f"""<div style='display: flex; gap: 1rem;'>{mean_display}{upper_display}{lower_display}</div>"""
        st.markdown(combined_display, unsafe_allow_html=True)

        if not earnings_m.empty:
            within_1sd = earnings_m[(earnings_m >= lower_bound) & (earnings_m <= upper_bound)]
            percent_within_1sd = (len(within_1sd) / len(earnings_m)) * 100 if len(earnings_m) > 0 else 0
            st.info(f"Percentage of players within ±1 SD: {percent_within_1sd:.2f}%")
        else:
            st.info("Percentage within ±1 SD cannot be calculated (no earnings data).")

        st.subheader("Empirical Cumulative Distribution Function (ECDF)")
        ecdf_y = (earnings.rank(method='first') / len(earnings)).values if len(earnings) > 0 else np.array([])
        ecdf_x = earnings.values
        median_val = earnings.median() if not earnings.empty else 0

        rank_info_for_ecdf = f"SGL Rank Range {sgl_rank_range[0]}–{sgl_rank_range[1]}" if use_rank_filter else f"Rank Bin {selected_bin if selected_bin else 'N/A'}"
        ecdf_title_text = f"ECDF of {plot_title_status} Prize Money for {years_str_for_plot_title} ({rank_info_for_ecdf})"

        if len(ecdf_x) > 0 and len(ecdf_y) > 0:
            fig_ecdf = px.line(x=ecdf_x, y=ecdf_y, labels={'x': earnings_column, 'y': 'Cumulative Proportion'}, title=ecdf_title_text)
            fig_ecdf.add_vline(x=median_val, line_dash="dash", line_color="red", annotation_text=f"Median: {median_val:,.0f}")
            fig_ecdf.update_layout(xaxis_tickformat=',', xaxis_tickangle=90)
            st.plotly_chart(fig_ecdf)
            st.success(f"Median {plot_title_status} Net Prize Money: {median_val:,.0f}")
        else:
            st.info("Not enough data to generate ECDF plot.")

    # --- Shortfall Comparison Section ---
    st.markdown("---")
    st.header("Shortfall Comparison")
    st.markdown(f"The **Expected Shortfall** (for Baseline Year 2025 using '{'Net Prize Money (2025 Adjusted)'}') is shown at the top of the page.")
    st.markdown("_All shortfall calculations include only players with >14 tournaments played and < £15M career prize money._")
    st.markdown("---")
    
    actual_shortfall_years_to_calc = []
    if not use_adjusted_earnings: 
        actual_shortfall_years_to_calc = selected_years
    else: 
        if selected_years:
            actual_shortfall_years_to_calc = [by - 1 for by in selected_years]
            available_actual_rank_years_in_df = df['Year'].unique()
            actual_shortfall_years_to_calc = [y for y in actual_shortfall_years_to_calc if y in available_actual_rank_years_in_df]

    # Global check flags for actual shortfall calculation
    snumtrn_exists_globally = 'snumtrn' in df.columns
    carprz_exists_globally = 'carprz' in df.columns

    if not actual_shortfall_years_to_calc:
        st.info("No Rank Years selected or available for Actual Shortfall calculation based on current filters.")
    else:
        st.subheader(f"Actual Shortfall Calculation (for Rank Year(s): {', '.join(map(str, sorted(list(set(actual_shortfall_years_to_calc)))))} using '{'Net Prize Money (Actual)'}')")
        
        # Display warnings for missing columns once before the loop for actual shortfall
        if not snumtrn_exists_globally:
            st.warning("Column 'snumtrn' not found. For Actual Shortfall, players cannot meet 'games played > 14' condition.")
        if not carprz_exists_globally:
            st.warning("Column 'carprz' not found. For Actual Shortfall, players cannot meet 'career prize < 15M' condition.")

        results_actual_shortfall = []
        for year_val in sorted(list(set(actual_shortfall_years_to_calc))):
            df_year_actual = df[df['Year'] == year_val].copy() # Use .copy()
            
            if df_year_actual.empty:
                results_actual_shortfall.append({
                    "Year": year_val, "Count_101_175": 0, "Shortfall_101_175": 0,
                    "Count_176_250": 0, "Shortfall_176_250": 0, "comment": "No data for this year"
                })
                continue

            base_mask_actual1 = (
                df_year_actual['sglrank'].between(101, 175)
                & (df_year_actual['Net Prize Money (Actual)'] < 200_000)
            )
            base_mask_actual2 = (
                df_year_actual['sglrank'].between(176, 250)
                & (df_year_actual['Net Prize Money (Actual)'] < 100_000)
            )

            if snumtrn_exists_globally:
                snumtrn_filter_actual = (df_year_actual['snumtrn'] > 14)
            else:
                snumtrn_filter_actual = pd.Series(False, index=df_year_actual.index)
            
            if carprz_exists_globally:
                carprz_filter_actual = (df_year_actual['carprz'] < 15_000_000)
            else:
                carprz_filter_actual = pd.Series(False, index=df_year_actual.index)

            mask_actual1 = base_mask_actual1 & snumtrn_filter_actual & carprz_filter_actual
            count_actual1 = mask_actual1.sum()
            shortfall_actual1 = (200_000 - df_year_actual.loc[mask_actual1, 'Net Prize Money (Actual)']).sum()

            mask_actual2 = base_mask_actual2 & snumtrn_filter_actual & carprz_filter_actual
            count_actual2 = mask_actual2.sum()
            shortfall_actual2 = (100_000 - df_year_actual.loc[mask_actual2, 'Net Prize Money (Actual)']).sum()
            
            results_actual_shortfall.append({
                "Year": year_val,
                "Count_101_175": count_actual1, "Shortfall_101_175": shortfall_actual1,
                "Count_176_250": count_actual2, "Shortfall_176_250": shortfall_actual2
            })

        if results_actual_shortfall:
            for row_data in results_actual_shortfall:
                st.markdown(f"**Rank Year: {int(row_data['Year'])}**")
                if row_data.get("comment"):
                    st.write(row_data["comment"])
                    st.markdown("---")
                    continue
                col1_act, col2_act = st.columns(2)
                col1_act.metric(label="Ranks 101–175: # below £200k (Actual)", value=int(row_data['Count_101_175']), delta=f"£{row_data['Shortfall_101_175']:,.0f} total actual shortfall", delta_color="off")
                col2_act.metric(label="Ranks 176–250: # below £100k (Actual)", value=int(row_data['Count_176_250']), delta=f"£{row_data['Shortfall_176_250']:,.0f} total actual shortfall", delta_color="off")
                st.markdown("---")
else:
    st.warning("Please upload the earnings data CSV file to proceed.")