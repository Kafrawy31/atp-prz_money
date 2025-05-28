import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde, iqr
import plotly.express as px
import plotly.graph_objects as go

st.title("ATP Player Earnings Analysis")

# --- File uploader ---
uploaded_file = st.file_uploader("Please upload the earnings data CSV file:", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["rankdate"])
    df['Year'] = df['rankdate'].dt.year
    df['Baseline Year'] = df['rankdate'].dt.year + 1

    sglrank_numeric = pd.to_numeric(df['sglrank'], errors='coerce')
    if not sglrank_numeric.isna().all() and sglrank_numeric.max() >=1 :
        max_rank = int(sglrank_numeric.max())
        # Adjust range to ensure labels can be generated up to max_rank
        bin_5_labels = [f"{i}-{i+4}" for i in range(1, max_rank + 1, 5)]
        bin_5_bins = range(1, (len(bin_5_labels) * 5) + 1 + 5, 5) # Ensure enough bins for labels

        bin_10_labels = [f"{i}-{i+9}" for i in range(1, max_rank + 1, 10)]
        bin_10_bins = range(1, (len(bin_10_labels) * 10) + 1 + 10, 10)


        df['Rank Bin (5s)'] = pd.cut(
            df['sglrank'],
            bins=bin_5_bins, 
            right=False,
            labels=bin_5_labels[:len(bin_5_bins)-1] if len(bin_5_labels) >= len(bin_5_bins)-1 else bin_5_labels, # guard labels length
            include_lowest=True,
        )
        df['Rank Bin (10s)'] = pd.cut(
            df['sglrank'],
            bins=bin_10_bins,
            right=False,
            labels=bin_10_labels[:len(bin_10_bins)-1] if len(bin_10_labels) >= len(bin_10_bins)-1 else bin_10_labels,
            include_lowest=True,
        )
    
        def sorted_bin_categories(series: pd.Series) -> list[str]:
            cats = series.dropna().astype(str).unique()
            cats = [c for c in cats if c != 'nan' and '-' in c] # Ensure valid bin format
            return sorted(cats, key=lambda lbl: int(lbl.split('-')[0]))

        if 'Rank Bin (5s)' in df.columns and not df['Rank Bin (5s)'].isna().all():
            valid_categories_5s = sorted_bin_categories(df['Rank Bin (5s)'])
            if valid_categories_5s:
                df['Rank Bin (5s)'] = df['Rank Bin (5s)'].astype('category').cat.set_categories(
                    valid_categories_5s, ordered=True)
        if 'Rank Bin (10s)' in df.columns and not df['Rank Bin (10s)'].isna().all():
            valid_categories_10s = sorted_bin_categories(df['Rank Bin (10s)'])
            if valid_categories_10s:
                df['Rank Bin (10s)'] = df['Rank Bin (10s)'].astype('category').cat.set_categories(
                    valid_categories_10s, ordered=True)
    else:
        st.warning("Single ranks ('sglrank') column has issues or no valid data; binning might be affected.")

    st.sidebar.title("Filters")
    use_adjusted_earnings = st.sidebar.checkbox("Show expected (adjusted) prize money for following year", value=False)
    year_column = 'Baseline Year' if use_adjusted_earnings else 'Year'
    
    if year_column in df.columns and df[year_column].nunique() > 0:
        years = sorted(df[year_column].dropna().unique())
        default_years = years
    else:
        years = []
        default_years = []
        st.sidebar.warning(f"No data available for '{year_column}'. Please check the column in your CSV.")
    selected_years = st.sidebar.multiselect(f"Select {year_column}(s)", options=years, default=default_years)
    earnings_column = 'Net Prize Money (2025 Adjusted)' if use_adjusted_earnings else 'Net Prize Money (Actual)'
    
    essential_cols = ['Net Prize Money (2025 Adjusted)', 'Net Prize Money (Actual)', 'sglrank']
    missing_essential_cols = [col for col in essential_cols if col not in df.columns]
    if missing_essential_cols:
        st.error(f"Essential column(s) not found: {', '.join(missing_essential_cols)}. Please check the uploaded file.")
        st.stop()
    
    shortfall_condition_cols = ['snumtrn', 'carprz']
    for col in shortfall_condition_cols:
        if col not in df.columns:
            st.warning(f"Column '{col}' not found. Shortfall calculations requiring this column will treat its condition as not met by any player.")

    st.subheader("Expected Shortfall Summary (for Baseline Year 2025, using Adjusted Earnings)")
    st.markdown("_Players must have >14 tournaments played and < £15M career prize money to be included in shortfall._")
    df_baseline_2025 = df[df['Baseline Year'] == 2025].copy()
    
    count0_expected, shortfall0_expected = 0, 0
    count1_expected, shortfall1_expected = 0, 0
    count2_expected, shortfall2_expected = 0, 0

    if not df_baseline_2025.empty:
        base_condition0_expected = (
            df_baseline_2025['sglrank'].between(51, 100)
            & (df_baseline_2025['Net Prize Money (2025 Adjusted)'] < 300_000) # Target £300k for 51-100
        )
        base_condition1_expected = (
            df_baseline_2025['sglrank'].between(101, 175)
            & (df_baseline_2025['Net Prize Money (2025 Adjusted)'] < 200_000)
        )
        base_condition2_expected = (
            df_baseline_2025['sglrank'].between(176, 250)
            & (df_baseline_2025['Net Prize Money (2025 Adjusted)'] < 100_000)
        )

        snumtrn_filter_expected = pd.Series(False, index=df_baseline_2025.index)
        if 'snumtrn' in df_baseline_2025.columns:
            snumtrn_filter_expected = (df_baseline_2025['snumtrn'] > 14)
        
        carprz_filter_expected = pd.Series(False, index=df_baseline_2025.index)
        if 'carprz' in df_baseline_2025.columns:
            carprz_filter_expected = (df_baseline_2025['carprz'] < 15_000_000)

        mask0_expected = base_condition0_expected & snumtrn_filter_expected & carprz_filter_expected
        count0_expected = mask0_expected.sum()
        shortfall0_expected = (300_000 - df_baseline_2025.loc[mask0_expected, 'Net Prize Money (2025 Adjusted)']).sum()

        mask1_expected = base_condition1_expected & snumtrn_filter_expected & carprz_filter_expected
        count1_expected = mask1_expected.sum()
        shortfall1_expected = (200_000 - df_baseline_2025.loc[mask1_expected, 'Net Prize Money (2025 Adjusted)']).sum()

        mask2_expected = base_condition2_expected & snumtrn_filter_expected & carprz_filter_expected
        count2_expected = mask2_expected.sum()
        shortfall2_expected = (100_000 - df_baseline_2025.loc[mask2_expected, 'Net Prize Money (2025 Adjusted)']).sum()
    
    c0_exp, c1_exp, c2_exp = st.columns(3)
    c0_exp.metric("Ranks 51–100: # below £300k", count0_expected, f"£{shortfall0_expected:,.0f} total expected shortfall")
    c1_exp.metric("Ranks 101–175: # below £200k", count1_expected, f"£{shortfall1_expected:,.0f} total expected shortfall")
    c2_exp.metric("Ranks 176–250: # below £100k", count2_expected, f"£{shortfall2_expected:,.0f} total expected shortfall")
    st.markdown("---")

    bin_type = st.sidebar.radio("Rank Bin Type", ["5s", "10s"])
    rank_bin_col_name = f'Rank Bin ({bin_type})'

    sgl_rank_min_df, sgl_rank_max_df = (int(sglrank_numeric.dropna().min()), int(sglrank_numeric.dropna().max())) if not sglrank_numeric.dropna().empty else (1,100)
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

    filtered = df.copy()
    if year_column in df.columns and selected_years:
         filtered = df[df[year_column].isin(selected_years)].copy()
    elif not selected_years and year_column in df.columns and df[year_column].nunique() > 0:
        st.info(f"No specific {year_column}(s) selected. Displaying data for all available {year_column}s.")
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
                selected_bin = st.sidebar.selectbox(f"Select Rank Bin ({bin_type})", options=bin_options)
                if selected_bin:
                    filtered = filtered[filtered[rank_bin_col_name] == selected_bin]
            else:
                st.sidebar.warning(f"No rank bins available for {bin_type} with current filters.")
        elif rank_bin_col_name in df.columns : 
             st.sidebar.warning(f"Rank bin column '{rank_bin_col_name}' has no categories for current selection.")

    if use_snumtrn_filter and snumtrn_range and 'snumtrn' in filtered.columns:
        filtered = filtered[(filtered['snumtrn'] >= snumtrn_range[0]) & (filtered['snumtrn'] <= snumtrn_range[1])]
    if use_carprz_filter and carprz_range and 'carprz' in filtered.columns:
        filtered = filtered[(filtered['carprz'] >= carprz_range[0]) & (filtered['carprz'] <= carprz_range[1])]
    if use_prize_money_filter and prize_range and earnings_column in filtered.columns:
        filtered = filtered[(filtered[earnings_column] >= prize_range[0]) & (filtered[earnings_column] <= prize_range[1])]
    
    earnings = pd.Series(dtype=float)
    if earnings_column in filtered.columns and not filtered[earnings_column].dropna().empty:
        earnings = filtered[earnings_column].dropna().sort_values().reset_index(drop=True)

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
        main_title_text = (f"**{title_year_prefix}** {years_display_string_main} • **SGL Rank Range:** {sgl_rank_range[0]}–{sgl_rank_range[1]} • **{title_earnings_suffix}**")
    else: 
        main_title_text = (f"**{title_year_prefix}** {years_display_string_main} • **Rank Bin:** {selected_bin if selected_bin else 'N/A'} ({bin_type}) • **{title_earnings_suffix}**")
    st.markdown(main_title_text)

    if earnings.empty:
        st.warning("No data available for the selected filter combination.")
    else:
        plot_title_status = "Adjusted" if use_adjusted_earnings else "Actual"
        years_str_for_plot_title = ""
        if selected_years:
            all_years_in_original_df_for_current_mode = df[year_column].dropna().unique()
            if len(all_years_in_original_df_for_current_mode) > 0 and len(selected_years) == len(all_years_in_original_df_for_current_mode) :
                years_str_for_plot_title = f"All {year_column}s"
            else:
                years_str_for_plot_title = f"{year_column}(s): {', '.join(map(str, sorted(selected_years)))}"
        elif year_column in df.columns and df[year_column].nunique() > 0: 
            years_str_for_plot_title = f"All Available {year_column}s"
        else:
            years_str_for_plot_title = f"No {year_column}s Data"

        st.subheader("Prize Money Distribution")
        counts_hist, bin_edges_hist = np.histogram(earnings, bins=15)
        max_count_hist = max(counts_hist) if len(counts_hist) > 0 else 0
        padded_max_hist = max_count_hist * 1.8 if max_count_hist > 0 else 10

        fig_hist = px.histogram(x=earnings, nbins=30, labels={'x': earnings_column, 'y': 'Count'}, text_auto=True)
        fig_hist.update_layout(
            title_text=f"Distribution of {plot_title_status} Prize Money for {years_str_for_plot_title}",
            yaxis=dict(range=[0, padded_max_hist]), yaxis_title="Count", xaxis_tickformat=',', xaxis_tickangle=90
        )
        fig_hist.update_traces(hovertemplate=f'{earnings_column}: %{{x}}<br>Count: %{{y}}<extra></extra>')
        st.plotly_chart(fig_hist)

        st.subheader(f"Density Curve of {plot_title_status} Net Prize Money")
        if earnings.empty or earnings.nunique() < 2:
            st.info("Not enough data points or variance to generate a density curve and its statistics.")
        else:
            earnings_m = earnings / 1e6
            median_val_m = earnings_m.median()
            abs_dev_from_median = (earnings_m - median_val_m).abs()
            mad_val_m = abs_dev_from_median.median()
            scaled_mad_m = mad_val_m * 1.4826 
            lower_bound_mad = max(median_val_m - scaled_mad_m, 0)
            upper_bound_mad = median_val_m + scaled_mad_m
            kde = gaussian_kde(earnings_m)
            x_range_kde = np.linspace(earnings_m.min(), earnings_m.max(), 200)
            y_kde = kde(x_range_kde)
            fig_kde = px.line(x=x_range_kde, y=y_kde, labels={'x': f'{earnings_column} (Millions)', 'y': 'Density'})
            fig_kde.update_layout(title_text=f"Density of {plot_title_status} Prize Money for {years_str_for_plot_title} (Median & Scaled MAD)", xaxis_tickformat=',.3f', xaxis_tickangle=90)
            fig_kde.add_vline(x=median_val_m, line_color="blue", line_dash="dot", annotation_text=f"Median: ${median_val_m:,.3f}m")
            fig_kde.add_vline(x=lower_bound_mad, line_color="purple", line_dash="dash", annotation_text=f"LMAD: ${lower_bound_mad:,.3f}m")
            fig_kde.add_vline(x=upper_bound_mad, line_color="purple", line_dash="dash", annotation_text=f"UMAD: ${upper_bound_mad:,.3f}m")
            fig_kde.update_traces(hovertemplate=f'{earnings_column} (Millions): %{{x:.3f}}<br>Density: %{{y:.3f}}<extra></extra>')
            st.plotly_chart(fig_kde)
            median_display = f"""<div style="background-color: #e0e0ff; color: #000080; border: 1px solid #b0b0e0; border-radius: 0.25rem; padding: 0.5rem 1rem; font-weight: bold;">Median: ${median_val_m * 1e6:,.0f}</div>"""
            upper_mad_display = f"""<div style="background-color: #f0e6ff; color: #4b0082; border: 1px solid #d8c0ff; border-radius: 0.25rem; padding: 0.5rem 1rem; font-weight: bold;">Median + Scaled MAD: ${upper_bound_mad * 1e6:,.0f}</div>"""
            lower_mad_display = f"""<div style="background-color: #f0e6ff; color: #4b0082; border: 1px solid #d8c0ff; border-radius: 0.25rem; padding: 0.5rem 1rem; font-weight: bold;">Median − Scaled MAD: ${lower_bound_mad * 1e6:,.0f}</div>"""
            combined_mad_display = f"""<div style='display: flex; gap: 1rem;'>{median_display}{upper_mad_display}{lower_mad_display}</div>"""
            st.markdown(combined_mad_display, unsafe_allow_html=True)
            within_mad_bounds = earnings_m[(earnings_m >= lower_bound_mad) & (earnings_m <= upper_bound_mad)]
            percent_within_mad_bounds = (len(within_mad_bounds) / len(earnings_m)) * 100 if len(earnings_m) > 0 else 0
            st.info(f"Percentage of players within Median ± Scaled MAD: {percent_within_mad_bounds:.2f}%")
            st.markdown("""**Note on Scaled MAD:** The Median Absolute Deviation (MAD) is a robust measure of spread. It is scaled here by a factor of ~1.4826 to make it comparable to the standard deviation for data that is approximately normally distributed. The interval 'Median ± Scaled MAD' provides a robust alternative to 'Mean ± Standard Deviation'.""")

        st.subheader("Empirical Cumulative Distribution Function (ECDF)")
        ecdf_y = (earnings.rank(method='first') / len(earnings)).values if len(earnings) > 0 else np.array([])
        ecdf_x = earnings.values
        median_val_ecdf = earnings.median() if not earnings.empty else 0
        rank_info_for_ecdf = f"SGL Rank Range {sgl_rank_range[0]}–{sgl_rank_range[1]}" if use_rank_filter else f"Rank Bin {selected_bin if selected_bin else 'N/A'}"
        ecdf_title_text = f"ECDF of {plot_title_status} Prize Money for {years_str_for_plot_title} ({rank_info_for_ecdf})"
        if len(ecdf_x) > 0 and len(ecdf_y) > 0:
            fig_ecdf = px.line(x=ecdf_x, y=ecdf_y, labels={'x': earnings_column, 'y': 'Cumulative Proportion'}, title=ecdf_title_text)
            fig_ecdf.add_vline(x=median_val_ecdf, line_dash="dash", line_color="red", annotation_text=f"Median: {median_val_ecdf:,.0f}")
            fig_ecdf.update_layout(xaxis_tickformat=',', xaxis_tickangle=90)
            st.plotly_chart(fig_ecdf)
            st.success(f"Median {plot_title_status} Net Prize Money: {median_val_ecdf:,.0f}")
        else:
            st.info("Not enough data to generate ECDF plot.")

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
            available_actual_rank_years_in_df = df['Year'].dropna().unique()
            actual_shortfall_years_to_calc = [y for y in actual_shortfall_years_to_calc if y in available_actual_rank_years_in_df]

    snumtrn_exists_globally = 'snumtrn' in df.columns
    carprz_exists_globally = 'carprz' in df.columns

    if not actual_shortfall_years_to_calc:
        st.info("No Rank Years selected or available for Actual Shortfall calculation based on current filters.")
    else:
        st.subheader(f"Actual Shortfall Analysis (for Rank Year(s): {', '.join(map(str, sorted(list(set(actual_shortfall_years_to_calc)))))} using '{'Net Prize Money (Actual)'}')")
        
        if not snumtrn_exists_globally:
            st.warning("Column 'snumtrn' not found. For Actual Shortfall, players cannot meet 'games played > 14' condition.")
        if not carprz_exists_globally:
            st.warning("Column 'carprz' not found. For Actual Shortfall, players cannot meet 'career prize < 15M' condition.")

        results_actual_shortfall = []
        for year_val in sorted(list(set(actual_shortfall_years_to_calc))):
            df_year_actual = df[df['Year'] == year_val].copy()
            
            count_actual0, shortfall_actual0 = 0, 0
            count_actual1, shortfall_actual1 = 0, 0
            count_actual2, shortfall_actual2 = 0, 0
            comment_for_year = None

            if df_year_actual.empty:
                comment_for_year = "No data for this year"
            else:
                base_mask_actual0 = (
                    df_year_actual['sglrank'].between(51, 100)
                    & (df_year_actual['Net Prize Money (Actual)'] < 300_000) # Target £350k
                )
                base_mask_actual1 = (
                    df_year_actual['sglrank'].between(101, 175)
                    & (df_year_actual['Net Prize Money (Actual)'] < 200_000)
                )
                base_mask_actual2 = (
                    df_year_actual['sglrank'].between(176, 250)
                    & (df_year_actual['Net Prize Money (Actual)'] < 100_000)
                )

                snumtrn_filter_actual = pd.Series(False, index=df_year_actual.index)
                if snumtrn_exists_globally and 'snumtrn' in df_year_actual.columns:
                    snumtrn_filter_actual = (df_year_actual['snumtrn'] > 14)
                
                carprz_filter_actual = pd.Series(False, index=df_year_actual.index)
                if carprz_exists_globally and 'carprz' in df_year_actual.columns:
                    carprz_filter_actual = (df_year_actual['carprz'] < 15_000_000)

                mask_actual0 = base_mask_actual0 & snumtrn_filter_actual & carprz_filter_actual
                count_actual0 = mask_actual0.sum()
                shortfall_actual0 = (300_000 - df_year_actual.loc[mask_actual0, 'Net Prize Money (Actual)']).sum()

                mask_actual1 = base_mask_actual1 & snumtrn_filter_actual & carprz_filter_actual
                count_actual1 = mask_actual1.sum()
                shortfall_actual1 = (200_000 - df_year_actual.loc[mask_actual1, 'Net Prize Money (Actual)']).sum()

                mask_actual2 = base_mask_actual2 & snumtrn_filter_actual & carprz_filter_actual
                count_actual2 = mask_actual2.sum()
                shortfall_actual2 = (100_000 - df_year_actual.loc[mask_actual2, 'Net Prize Money (Actual)']).sum()
            
            results_actual_shortfall.append({
                "Year": year_val,
                "Count_51_100": count_actual0, "Shortfall_51_100": shortfall_actual0,
                "Count_101_175": count_actual1, "Shortfall_101_175": shortfall_actual1,
                "Count_176_250": count_actual2, "Shortfall_176_250": shortfall_actual2,
                "comment": comment_for_year
            })

        if results_actual_shortfall:
            actual_shortfall_df = pd.DataFrame(results_actual_shortfall)
            actual_shortfall_df = actual_shortfall_df.sort_values(by='Year')
            plot_df = actual_shortfall_df[actual_shortfall_df['comment'].isna()].copy()

            if not plot_df.empty and plot_df['Year'].nunique() > 0 : # Require at least one year of data to plot
                st.markdown("---")
                st.subheader("Trend of Actual Shortfall Players by Year")
                fig_count_trend = px.line(
                    plot_df, x='Year', y=['Count_51_100', 'Count_101_175', 'Count_176_250'],
                    labels={'value': 'Number of Players', 'Year': 'Rank Year'}, markers=True,
                    title="Number of Players with Actual Shortfall by Rank Year"
                )
                count_trace_names = {
                    'Count_51_100': 'Ranks 51-100 (Target < £300k)',
                    'Count_101_175': 'Ranks 101-175 (Target < £200k)',
                    'Count_176_250': 'Ranks 176-250 (Target < £100k)'
                }
                fig_count_trend.for_each_trace(lambda t: t.update(name=count_trace_names.get(t.name, t.name)))
                fig_count_trend.update_layout(legend_title_text='Player Groups')
                st.plotly_chart(fig_count_trend)

                st.subheader("Trend of Actual Shortfall Amount by Year")
                fig_amount_trend = px.line(
                    plot_df, x='Year', y=['Shortfall_51_100', 'Shortfall_101_175', 'Shortfall_176_250'],
                    labels={'value': 'Total Shortfall Amount (£)', 'Year': 'Rank Year'}, markers=True,
                    title="Actual Shortfall Amount by Rank Year"
                )
                amount_trace_names = {
                    'Shortfall_51_100': 'Ranks 51-100 (Target < £300k)',
                    'Shortfall_101_175': 'Ranks 101-175 (Target < £200k)',
                    'Shortfall_176_250': 'Ranks 176-250 (Target < £100k)'
                }
                fig_amount_trend.for_each_trace(lambda t: t.update(name=amount_trace_names.get(t.name, t.name)))
                fig_amount_trend.update_layout(legend_title_text='Shortfall Amounts', yaxis_tickformat=',.0f')
                st.plotly_chart(fig_amount_trend)
            elif not actual_shortfall_df.empty :
                 st.info("Not enough yearly data points without comments to plot trends for actual shortfall.")

            st.markdown("---")
            st.subheader("Detailed Actual Shortfall by Year:")
            for _, row_data in actual_shortfall_df.iterrows():
                st.markdown(f"**Rank Year: {int(row_data['Year'])}**")
                if row_data["comment"]:
                    st.write(row_data["comment"])
                    st.markdown("---")
                    continue
                
                col0_act, col1_act, col2_act = st.columns(3)
                col0_act.metric(label="Ranks 51–100: # below £300k (Actual)", value=int(row_data['Count_51_100']), delta=f"£{row_data['Shortfall_51_100']:,.0f} total actual shortfall", delta_color="off")
                col1_act.metric(label="Ranks 101–175: # below £200k (Actual)", value=int(row_data['Count_101_175']), delta=f"£{row_data['Shortfall_101_175']:,.0f} total actual shortfall", delta_color="off")
                col2_act.metric(label="Ranks 176–250: # below £100k (Actual)", value=int(row_data['Count_176_250']), delta=f"£{row_data['Shortfall_176_250']:,.0f} total actual shortfall", delta_color="off")
                st.markdown("---")
        else:
            st.info("No data processed for Actual Shortfall calculation.")
else:
    st.warning("Please upload the earnings data CSV file to proceed.")