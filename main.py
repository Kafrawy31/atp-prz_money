import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde # iqr is not used, can be removed if not needed elsewhere
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
    sgl_rank_min_df_default, sgl_rank_max_df_default = (1, 250) # Sensible defaults
    if not sglrank_numeric.isna().all() and sglrank_numeric.max() >= 1:
        sgl_rank_min_df_default = int(sglrank_numeric.dropna().min())
        sgl_rank_max_df_default = int(sglrank_numeric.dropna().max())
    else:
        st.warning("Single ranks ('sglrank') column has issues, is empty, or contains no valid rank data. Some features might be affected. Defaulting rank filter range.")

    # --- Initialize Session State for Rank Filters ---
    if 'sgl_rank_min_val' not in st.session_state:
        st.session_state.sgl_rank_min_val = sgl_rank_min_df_default
    if 'sgl_rank_max_val' not in st.session_state:
        st.session_state.sgl_rank_max_val = sgl_rank_max_df_default
    if 'use_rank_filter_val' not in st.session_state:
        st.session_state.use_rank_filter_val = False # Default to not pre-applying a filter
    if 'rank_preset_key' not in st.session_state: # For the radio button's own state
        st.session_state.rank_preset_key = "Custom"


    st.sidebar.title("Filters")

    # --- Dynamic Guarantee Value Inputs (in Thousands of Dollars) ---

    use_adjusted_earnings = st.sidebar.checkbox("Show expected (adjusted) prize money for following year", value=False)
    year_column = 'Baseline Year' if use_adjusted_earnings else 'Year'
    
    if year_column in df.columns and df[year_column].nunique() > 0:
        years = sorted(df[year_column].dropna().unique())
        default_years = years
    else:
        years = []
        default_years = []
        st.sidebar.warning(f"No data available for '{year_column}'.")
    selected_years = st.sidebar.multiselect(f"Select {year_column}(s)", options=years, default=default_years)
    earnings_column = 'Net Prize Money (2025 Adjusted)' if use_adjusted_earnings else 'Net Prize Money (Actual)'
    
    essential_cols = ['Net Prize Money (2025 Adjusted)', 'Net Prize Money (Actual)', 'sglrank']
    missing_essential_cols = [col for col in essential_cols if col not in df.columns]
    if missing_essential_cols:
        st.error(f"Essential column(s) not found: {', '.join(missing_essential_cols)}.")
        st.stop()
    
    exposure_condition_cols = ['snumtrn', 'carprz']
    for col in exposure_condition_cols:
        if col not in df.columns:
            st.warning(f"Column '{col}' not found. Exposure calculations might be affected.")

    # --- Rank Range Selection Logic ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Rank Range Selection")

    rank_presets_options = {
        "Custom": None,
        "Ranks 1-100": (1, 100),
        "Ranks 101-175": (101, 175),
        "Ranks 176-250": (176, 250)
    }

    st.sidebar.subheader("Set Guarantee Levels ($k)")
    guarantee_51_100_k = st.sidebar.number_input("Ranks 51-100 Guarantee ($k)", value=300, min_value=0, step=10, format="%d", key="g51_100k_usd")
    guarantee_101_175_k = st.sidebar.number_input("Ranks 101-175 Guarantee ($k)", value=200, min_value=0, step=10, format="%d", key="g101_175k_usd")
    guarantee_176_250_k = st.sidebar.number_input("Ranks 176-250 Guarantee ($k)", value=100, min_value=0, step=10, format="%d", key="g176_250k_usd")

    guarantee_51_100 = guarantee_51_100_k * 1000
    guarantee_101_175 = guarantee_101_175_k * 1000
    guarantee_176_250 = guarantee_176_250_k * 1000

    guarantee_map_dynamic = [
        {"min_r": 1,   "max_r": 100, "value": guarantee_51_100, "label_short": "G'tee (1-100)", "label_full": f"Guarantee (Ranks 1-100: ${guarantee_51_100:,.0f})"}, # Adjusted for common preset
        {"min_r": 101, "max_r": 175, "value": guarantee_101_175, "label_short": "G'tee (101-175)", "label_full": f"Guarantee (Ranks 101-175: ${guarantee_101_175:,.0f})"},
        {"min_r": 176, "max_r": 250, "value": guarantee_176_250, "label_short": "G'tee (176-250)", "label_full": f"Guarantee (Ranks 176-250: ${guarantee_176_250:,.0f})"},
    ] # Note: Included a 1-100 mapping assuming it might use the 51-100 guarantee. Adjust if different.
    
    def on_preset_change():
        preset_name = st.session_state.rank_preset_radio # Get the selected preset name
        if preset_name != "Custom":
            min_r, max_r = rank_presets_options[preset_name]
            st.session_state.sgl_rank_min_val = min_r
            st.session_state.sgl_rank_max_val = max_r
            st.session_state.use_rank_filter_val = True
        # If "Custom" is selected, we don't change anything here; user uses manual controls.
    
    # Determine the current index for the radio button based on session state
    # This is to make the radio button reflect the actual filter state if possible
    current_filter_is_preset = "Custom"
    if st.session_state.use_rank_filter_val:
        current_tuple = (st.session_state.sgl_rank_min_val, st.session_state.sgl_rank_max_val)
        for name, prange in rank_presets_options.items():
            if prange == current_tuple:
                current_filter_is_preset = name
                break
    
    st.sidebar.radio(
        "Quick Presets:",
        options=list(rank_presets_options.keys()),
        key='rank_preset_radio', # Key for the radio widget itself
        on_change=on_preset_change,
        index=list(rank_presets_options.keys()).index(current_filter_is_preset) # Set default index
    )

    st.session_state.use_rank_filter_val = st.sidebar.checkbox(
        "Filter by Rank Range (enable for presets or custom)",
        key='use_rank_filter_val_cb', # Using the actual session state var as key
        value=st.session_state.use_rank_filter_val
    )
    use_rank_filter = st.session_state.use_rank_filter_val # Main variable to use

    sgl_rank_range_applied = (sgl_rank_min_df_default, sgl_rank_max_df_default) # Default applied range

    if use_rank_filter:
        st.sidebar.write("Define Rank Range:")
        # Update session state directly from number inputs
        new_min_rank = st.sidebar.number_input(
            "Min SGL Rank",
            min_value=1, max_value=sgl_rank_max_df_default + 500, # Allow some flexibility
            value=st.session_state.sgl_rank_min_val,
            key='sgl_rank_min_input_widget' # Distinct key for widget
        )
        new_max_rank = st.sidebar.number_input(
            "Max SGL Rank",
            min_value=1, max_value=sgl_rank_max_df_default + 500,
            value=st.session_state.sgl_rank_max_val,
            key='sgl_rank_max_input_widget' # Distinct key for widget
        )
        # If manual input changes, update session state and potentially set preset to "Custom"
        if new_min_rank != st.session_state.sgl_rank_min_val or new_max_rank != st.session_state.sgl_rank_max_val:
            st.session_state.sgl_rank_min_val = new_min_rank
            st.session_state.sgl_rank_max_val = new_max_rank
            st.session_state.rank_preset_key = "Custom" # Reflect that it's now custom
            # No st.experimental_rerun() needed, happens naturally

        if st.session_state.sgl_rank_min_val > st.session_state.sgl_rank_max_val:
            st.session_state.sgl_rank_max_val = st.session_state.sgl_rank_min_val
            # No warning needed here as it auto-corrects, or let Streamlit handle if min > max for number_input
        
        sgl_rank_range_applied = (st.session_state.sgl_rank_min_val, st.session_state.sgl_rank_max_val)
    
    # --- Other Filters ---
    use_snumtrn_filter = st.sidebar.checkbox("Use Tournament Filter (snumtrn)")
    # ... (rest of snumtrn, carprz, prize_money filters remain the same) ...
    snumtrn_range = None
    if use_snumtrn_filter and 'snumtrn' in df.columns and not df['snumtrn'].dropna().empty:
        snumtrn_min, snumtrn_max = int(df['snumtrn'].dropna().min()), int(df['snumtrn'].dropna().max())
        snumtrn_range = st.sidebar.slider("Select number of tournaments Range", min_value=snumtrn_min, max_value=snumtrn_max, value=(snumtrn_min, snumtrn_max))
    elif use_snumtrn_filter:
        st.sidebar.warning("'snumtrn' column not found or empty; cannot apply tournament filter.")

    use_carprz_filter = st.sidebar.checkbox("Use career prize Filter (carprz)")
    carprz_range = None
    if use_carprz_filter and 'carprz' in df.columns and not df['carprz'].dropna().empty:
        carprz_min_val, carprz_max_val = int(df['carprz'].dropna().min()), int(df['carprz'].dropna().max()) # Avoid conflict
        st.sidebar.write("Enter career prize Range ($):") 
        carprz_min_input = st.sidebar.number_input("Min career prize ($)", value=carprz_min_val, min_value=carprz_min_val, max_value=carprz_max_val, format="%d")
        carprz_max_input = st.sidebar.number_input("Max career prize ($)", value=carprz_max_val, min_value=carprz_min_val, max_value=carprz_max_val, format="%d")
        carprz_range = (carprz_min_input, carprz_max_input)
    elif use_carprz_filter:
        st.sidebar.warning("'carprz' column not found or empty; cannot apply career prize filter.")

    use_prize_money_filter = st.sidebar.checkbox("Use Prize Money Filter")
    prize_range = None
    if use_prize_money_filter and earnings_column in df.columns:
        if not df[earnings_column].dropna().empty:
            prize_min_val, prize_max_val = int(df[earnings_column].dropna().min()), int(df[earnings_column].dropna().max()) # Avoid conflict
            st.sidebar.write("Enter Prize Money Range ($):") 
            prize_min_input = st.sidebar.number_input("Min Prize Money ($)", value=prize_min_val, min_value=prize_min_val, max_value=prize_max_val, format="%d")
            prize_max_input = st.sidebar.number_input("Max Prize Money ($)", value=prize_max_val, min_value=prize_min_val, max_value=prize_max_val, format="%d")
            prize_range = (prize_min_input, prize_max_input)
        else:
            st.sidebar.warning(f"No data in '{earnings_column}' for prize money filter.")
            use_prize_money_filter = False
    elif use_prize_money_filter:
        st.sidebar.warning(f"'{earnings_column}' column not found for prize money filter.")


    # --- Expected Exposure Summary (uses full guarantee values) ---
    st.subheader("Expected Exposure Summary (for Baseline Year 2025, using Adjusted Earnings)")
    st.markdown(f"Players must have >14 tournaments played and < $15M in total career earnings. Note, guarantee thresholds are user configurable ")
    df_baseline_2025 = df[df['Baseline Year'] == 2025].copy()
    
    count0_expected, exposure0_expected = 0, 0 
    count1_expected, exposure1_expected = 0, 0 
    count2_expected, exposure2_expected = 0, 0 

    if not df_baseline_2025.empty:
        # Assuming guarantee_51_100, guarantee_101_175, guarantee_176_250 are the ones for these fixed bands
        base_condition0_expected = (df_baseline_2025['sglrank'].between(51, 100) & (df_baseline_2025['Net Prize Money (2025 Adjusted)'] < guarantee_51_100))
        base_condition1_expected = (df_baseline_2025['sglrank'].between(101, 175) & (df_baseline_2025['Net Prize Money (2025 Adjusted)'] < guarantee_101_175))
        base_condition2_expected = (df_baseline_2025['sglrank'].between(176, 250) & (df_baseline_2025['Net Prize Money (2025 Adjusted)'] < guarantee_176_250))

        snumtrn_filter_expected = pd.Series(True, index=df_baseline_2025.index) # Default to true if col missing
        if 'snumtrn' in df_baseline_2025.columns:
            snumtrn_filter_expected = (df_baseline_2025['snumtrn'] > 14)
        
        carprz_filter_expected = pd.Series(True, index=df_baseline_2025.index) # Default to true if col missing
        if 'carprz' in df_baseline_2025.columns:
            carprz_filter_expected = (df_baseline_2025['carprz'] < 15_000_000) 

        mask0_expected = base_condition0_expected & snumtrn_filter_expected & carprz_filter_expected
        count0_expected = mask0_expected.sum()
        if count0_expected > 0: exposure0_expected = (guarantee_51_100 - df_baseline_2025.loc[mask0_expected, 'Net Prize Money (2025 Adjusted)']).sum()

        mask1_expected = base_condition1_expected & snumtrn_filter_expected & carprz_filter_expected
        count1_expected = mask1_expected.sum()
        if count1_expected > 0: exposure1_expected = (guarantee_101_175 - df_baseline_2025.loc[mask1_expected, 'Net Prize Money (2025 Adjusted)']).sum()

        mask2_expected = base_condition2_expected & snumtrn_filter_expected & carprz_filter_expected
        count2_expected = mask2_expected.sum()
        if count2_expected > 0: exposure2_expected = (guarantee_176_250 - df_baseline_2025.loc[mask2_expected, 'Net Prize Money (2025 Adjusted)']).sum()
    
    c0_exp, c1_exp, c2_exp = st.columns(3)
    c0_exp.metric(f"Ranks 51–100: # below ${guarantee_51_100:,.0f}", count0_expected, f"${exposure0_expected:,.0f} total expected exposure")
    c1_exp.metric(f"Ranks 101–175: # below ${guarantee_101_175:,.0f}", count1_expected, f"${exposure1_expected:,.0f} total expected exposure")
    c2_exp.metric(f"Ranks 176–250: # below ${guarantee_176_250:,.0f}", count2_expected, f"${exposure2_expected:,.0f} total expected exposure")
    st.markdown("---")

    # --- Data Filtering ---
    filtered = df.copy()
    if year_column in df.columns and selected_years:
        filtered = df[df[year_column].isin(selected_years)].copy()
    # ... (other year selection messages) ...

    if use_rank_filter: # Applied rank range
        if 'sglrank' in filtered.columns:
            filtered = filtered[(filtered['sglrank'] >= sgl_rank_range_applied[0]) & (filtered['sglrank'] <= sgl_rank_range_applied[1])]
    
    # ... (snumtrn, carprz, prize_money filters applied to 'filtered' DataFrame) ...
    if use_snumtrn_filter and snumtrn_range and 'snumtrn' in filtered.columns:
        filtered = filtered[(filtered['snumtrn'] >= snumtrn_range[0]) & (filtered['snumtrn'] <= snumtrn_range[1])]
    if use_carprz_filter and carprz_range and 'carprz' in filtered.columns:
        filtered = filtered[(filtered['carprz'] >= carprz_range[0]) & (filtered['carprz'] <= carprz_range[1])]
    if use_prize_money_filter and prize_range and earnings_column in filtered.columns:
        filtered = filtered[(filtered[earnings_column] >= prize_range[0]) & (filtered[earnings_column] <= prize_range[1])]

    earnings = pd.Series(dtype=float)
    if earnings_column in filtered.columns and not filtered[earnings_column].dropna().empty:
        earnings = filtered[earnings_column].dropna().sort_values().reset_index(drop=True)

    # --- Main Title ---
    title_year_prefix = "Baseline Year(s):" if use_adjusted_earnings else "Year(s):"
    # ... (years_display_string_main logic) ...
    years_display_string_main = "None Selected"
    if selected_years:
        all_available_years_for_mode = df[year_column].dropna().unique()
        if len(all_available_years_for_mode) > 0 and len(selected_years) == len(all_available_years_for_mode):
            years_display_string_main = f"All ({year_column})"
        else:
            years_display_string_main = ', '.join(map(str, sorted(selected_years)))
    elif year_column in df.columns and df[year_column].nunique() > 0 :
            years_display_string_main = f"All ({year_column} - Default)"


    if use_rank_filter:
        rank_display_string_main = f"SGL Rank Range: {sgl_rank_range_applied[0]}–{sgl_rank_range_applied[1]}"
    else:
        rank_display_string_main = "SGL Rank Range: All Ranks (default)"
    title_earnings_suffix = "Expected (Adjusted) Earnings" if use_adjusted_earnings else "Actual Earnings"
    main_title_text = (f"**{title_year_prefix}** {years_display_string_main} • **{rank_display_string_main}** • **{title_earnings_suffix}**")
    st.markdown(main_title_text)


    # --- Helper function for more lenient guarantee matching ---
    def get_relevant_guarantee_info_for_display(user_min_r, user_max_r, is_filter_active, current_filtered_df, guarantee_map):
        # If a specific rank filter is active, prioritize matching that range
        if is_filter_active:
            # Priority 1: User range is fully contained within a guarantee band
            for g_info in guarantee_map:
                # Skip the generic 1-100 if a more specific 51-100 is also possible for same value
                if g_info["min_r"] == 1 and g_info["max_r"] == 100 and \
                   any(g2["min_r"] == 51 and g2["max_r"] == 100 and g2["value"] == g_info["value"] for g2 in guarantee_map):
                    if user_min_r >= 51: # only consider 1-100 if user range starts low
                        pass # let 51-100 take precedence if user_min_r is higher
                    elif user_min_r >= g_info["min_r"] and user_max_r <= g_info["max_r"]:
                         return g_info # e.g. user selects 1-50, matches 1-100
                elif user_min_r >= g_info["min_r"] and user_max_r <= g_info["max_r"]:
                    return g_info

            # Priority 2: Midpoint of user range falls within a guarantee band, and there's overlap
            user_mid_r = (user_min_r + user_max_r) / 2
            for g_info in guarantee_map:
                overlap_min = max(user_min_r, g_info["min_r"])
                overlap_max = min(user_max_r, g_info["max_r"])
                if overlap_min <= overlap_max: # Overlap exists
                    if g_info["min_r"] <= user_mid_r <= g_info["max_r"]:
                        return g_info 
            
            # Priority 3: Largest overlap if no other match
            largest_overlap_amount = 0
            most_overlapped_g_info = None
            for g_info in guarantee_map:
                overlap_start = max(user_min_r, g_info["min_r"])
                overlap_end = min(user_max_r, g_info["max_r"])
                overlap_length = overlap_end - overlap_start
                if overlap_length > 0: # Ensure positive overlap
                    # Prefer bands that are not excessively wider than user's range
                    band_width_ratio = (g_info["max_r"] - g_info["min_r"] + 1) / (user_max_r - user_min_r + 1)
                    if overlap_length > largest_overlap_amount and band_width_ratio < 5: # Heuristic
                        largest_overlap_amount = overlap_length
                        most_overlapped_g_info = g_info
            if most_overlapped_g_info:
                return most_overlapped_g_info
            return None
        
        # Fallback: if no specific rank filter is active, use median of currently filtered data
        else:
            if not current_filtered_df.empty and 'sglrank' in current_filtered_df.columns and current_filtered_df['sglrank'].nunique() > 0:
                data_median_rank_for_plot = current_filtered_df['sglrank'].median()
                for g_info_fallback in guarantee_map:
                    if g_info_fallback["min_r"] <= data_median_rank_for_plot <= g_info_fallback["max_r"]:
                        data_min_rank_for_plot = current_filtered_df['sglrank'].min()
                        data_max_rank_for_plot = current_filtered_df['sglrank'].max()
                        # Heuristic: data span should be somewhat related to band width
                        if (data_max_rank_for_plot - data_min_rank_for_plot) < ((g_info_fallback["max_r"] - g_info_fallback["min_r"]) + 75):
                            return g_info_fallback
            return None

    if earnings.empty:
        st.warning("No data available for the selected filter combination.")
    else:
        # ... (plot_title_status, years_str_for_plot_title logic remains same) ...
        plot_title_status = "Adjusted" if use_adjusted_earnings else "Actual"
        # ... (years_str_for_plot_title setup as before) ...
        years_str_for_plot_title = "Selected Year(s)" # Placeholder, use your existing logic
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


        # --- Histogram (remains largely the same) ---
        st.subheader("Prize Money Distribution")
        # ... (histogram code as before, ensuring $ sign is used) ...
        if earnings.empty:
            st.warning("No earnings data to plot for histogram.")
        else:
            actual_min_earning = earnings.min()
            actual_max_earning = earnings.max()
            hist_range_x_min = 0
            if actual_min_earning < 0: hist_range_x_min = actual_min_earning
            hist_range_x_max = actual_max_earning
            if actual_max_earning <= hist_range_x_min: hist_range_x_max = hist_range_x_min + (100000 if hist_range_x_min == 0 else abs(hist_range_x_min * 0.5) + 1)
            num_bins = 30 
            counts_hist, bin_edges_hist = np.histogram(earnings, bins=num_bins, range=(hist_range_x_min, hist_range_x_max))
            bin_labels = [f"${bin_edges_hist[i]:,.0f} - ${bin_edges_hist[i+1]:,.0f}" for i in range(len(counts_hist))]
            hist_plot_data = pd.DataFrame({'Bin_Range_Label': bin_labels, 'Player_Count': counts_hist, 'Bin_Lower_Edge': bin_edges_hist[:-1], 'Bin_Upper_Edge': bin_edges_hist[1:]})
            max_count_hist = max(counts_hist) if len(counts_hist) > 0 else 0
            padded_max_hist = max_count_hist * 1.8 if max_count_hist > 0 else 10
            fig_hist = px.bar(hist_plot_data, x='Bin_Range_Label', y='Player_Count', text='Player_Count', custom_data=['Bin_Lower_Edge', 'Bin_Upper_Edge'])
            fig_hist.update_traces(textposition='outside')
            fig_hist.update_layout(title_text=f"Distribution of {plot_title_status} Prize Money for {years_str_for_plot_title}", yaxis=dict(range=[0, padded_max_hist], title_text="Number of Players"), xaxis=dict(title_text=f"{earnings_column} Bins", tickangle=-90, type='category'), bargap=0.1)
            fig_hist.update_traces(hovertemplate=(f"<b>{earnings_column} Range:</b> $%{{customdata[0]:,.0f}} - $%{{customdata[1]:,.0f}}<br>" "<b>Number of Players:</b> %{y}<extra></extra>"))
            st.plotly_chart(fig_hist)

        # --- Density Curve (KDE) ---
        st.subheader(f"Density Curve of {plot_title_status} Net Prize Money")
        if earnings.empty or earnings.nunique() < 2: 
            st.info("Not enough data points or variance to generate a density curve.")
        else:
            earnings_m = earnings / 1e6 
            median_val_m = earnings_m.median()
            abs_dev_from_median = (earnings_m - median_val_m).abs()
            mad_val_m = abs_dev_from_median.median()
            scaled_mad_m = mad_val_m * 1.4826 
            lower_bound_mad = max(median_val_m - scaled_mad_m, 0) 
            upper_bound_mad = median_val_m + scaled_mad_m

            kde = gaussian_kde(earnings_m)
            min_earnings_m, max_earnings_m = earnings_m.min(), earnings_m.max()
            # ... (x_range_kde, y_kde calculation as before) ...
            if min_earnings_m == max_earnings_m:
                x_range_kde = np.linspace(min_earnings_m - 0.1 * abs(min_earnings_m) if min_earnings_m != 0 else -0.1, 
                                            max_earnings_m + 0.1 * abs(max_earnings_m) if max_earnings_m != 0 else 0.1, 200)
            else:
                x_range_kde = np.linspace(min_earnings_m, max_earnings_m, 200)
            y_kde = kde(x_range_kde)

            
            fig_kde = px.line(x=x_range_kde, y=y_kde, labels={'x': f'{earnings_column} (Millions $)', 'y': 'Density'})
            fig_kde.update_layout(title_text=f"Density of {plot_title_status} Prize Money ({years_str_for_plot_title}, Median & Scaled MAD)", xaxis_tickformat='$,.3f', xaxis_tickangle=90) # Ensure $
            fig_kde.add_vline(x=median_val_m, line_color="blue", line_dash="dot", annotation_text=f"${median_val_m:,.3f}m")
            fig_kde.add_vline(x=lower_bound_mad, line_color="purple", line_dash="dash", annotation_text=f"${lower_bound_mad:,.3f}m") 
            fig_kde.add_vline(x=upper_bound_mad, line_color="purple", line_dash="dash", annotation_text=f"${upper_bound_mad:,.3f}m")
            
            # Use the helper function to determine which guarantee line to show on the plot
            plot_display_guarantee_info = get_relevant_guarantee_info_for_display(
                sgl_rank_range_applied[0], sgl_rank_range_applied[1], use_rank_filter, filtered, guarantee_map_dynamic
            )
            
            if plot_display_guarantee_info:
                guarantee_val_m = plot_display_guarantee_info["value"] / 1_000_000 
                annotation_text_for_line = f"{plot_display_guarantee_info['label_short']}: ${guarantee_val_m:,.3f}m"
                # ... (logic to add vline for plot_display_guarantee_info to fig_kde as before) ...
                plot_x_min_kde, plot_x_max_kde = min_earnings_m, max_earnings_m # Use KDE specific range
                if plot_x_min_kde == plot_x_max_kde : 
                    plot_x_min_kde = plot_x_min_kde - 0.1 if plot_x_min_kde !=0 else -0.1
                    plot_x_max_kde = plot_x_max_kde + 0.1 if plot_x_max_kde !=0 else 0.1
                if (plot_x_min_kde - 0.05 * (plot_x_max_kde - plot_x_min_kde)) <= guarantee_val_m <= (plot_x_max_kde + 0.05 * (plot_x_max_kde - plot_x_min_kde)):
                    fig_kde.add_vline(x=guarantee_val_m, line_color="seagreen", line_dash="longdashdot", annotation_text=annotation_text_for_line, annotation_position="bottom right")

            fig_kde.update_traces(hovertemplate=f'{earnings_column} (Millions $): %{{x:.3f}}<br>Density: %{{y:.3f}}<extra></extra>')
            st.plotly_chart(fig_kde, key="kde_density_plot_exposure_stats")

            # --- HTML Stat Boxes (Median, MAD, General Guarantee) ---
            median_display_html = f"""<div style="background-color: #e0e0ff; color: #000080; border: 1px solid #b0b0e0; border-radius: 0.25rem; padding: 0.5rem 1rem; font-weight: bold;">Median: ${median_val_m * 1e6:,.0f}</div>"""
            upper_mad_display_html = f"""<div style="background-color: #f0e6ff; color: #4b0082; border: 1px solid #d8c0ff; border-radius: 0.25rem; padding: 0.5rem 1rem; font-weight: bold;">Upper Bound: ${upper_bound_mad * 1e6:,.0f}</div>"""
            lower_mad_display_html = f"""<div style="background-color: #f0e6ff; color: #4b0082; border: 1px solid #d8c0ff; border-radius: 0.25rem; padding: 0.5rem 1rem; font-weight: bold;">Lower Bound: ${lower_bound_mad * 1e6:,.0f}</div>"""
            
            general_guarantee_display_html = ""
            if plot_display_guarantee_info: 
                general_guarantee_text_for_box = f"{plot_display_guarantee_info['label_full']}" 
                general_guarantee_display_html = f"""<div style="background-color: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; border-radius: 0.25rem; padding: 0.5rem 1rem; font-weight: bold;">{general_guarantee_text_for_box}</div>"""

            current_display_elements = [lower_mad_display_html, median_display_html, upper_mad_display_html]
            if general_guarantee_display_html:
                current_display_elements.append(general_guarantee_display_html)
            
            st.markdown(f"""<div style='display: flex; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 1rem;'>{''.join(current_display_elements)}</div>""", unsafe_allow_html=True)

            st.info(f"Percentage of players (in current filtered data) within Median ± Scaled MAD: {(len(earnings_m[(earnings_m >= lower_bound_mad) & (earnings_m <= upper_bound_mad)]) / len(earnings_m) * 100 if len(earnings_m) > 0 else 0):.2f}%")
            # ... (MAD note) ...
            st.markdown("""**Note on Scaled MAD:** The Median Absolute Deviation (MAD) is a robust measure of spread. It is scaled here by a factor of ~1.4826 to make it comparable to the standard deviation for data that is approximately normally distributed. The interval 'Median ± Scaled MAD' provides a robust alternative to 'Mean ± Standard Deviation'.""")


        # --- ECDF Plot ---
        st.subheader("Empirical Cumulative Distribution Function (ECDF)")
        # ... (ECDF setup as before) ...
        ecdf_y_values = np.array([])
        ecdf_x_values = np.array([])
        median_val_ecdf = 0
        if not earnings.empty:
            ecdf_x_values = earnings.values 
            ecdf_y_values = (np.arange(1, len(earnings) + 1) / len(earnings)) 
            median_val_ecdf = earnings.median()

        if use_rank_filter: rank_info_for_ecdf = f"SGL Rank Range {sgl_rank_range_applied[0]}–{sgl_rank_range_applied[1]}"
        else: rank_info_for_ecdf = "All SGL Ranks (default)"
        ecdf_title_text = f"ECDF of {plot_title_status} Prize Money for {years_str_for_plot_title} ({rank_info_for_ecdf})"
        
        if len(ecdf_x_values) > 0 and len(ecdf_y_values) > 0:
            fig_ecdf = px.line(x=ecdf_x_values, y=ecdf_y_values, labels={'x': f'{earnings_column} ($)', 'y': 'Cumulative Proportion'}, title=ecdf_title_text)
            fig_ecdf.add_vline(x=median_val_ecdf, line_dash="dash", line_color="red", annotation_text=f"Median: ${median_val_ecdf:,.0f}")

            ecdf_plot_guarantee_info = get_relevant_guarantee_info_for_display(
                sgl_rank_range_applied[0], sgl_rank_range_applied[1], use_rank_filter, filtered, guarantee_map_dynamic
            )
            if ecdf_plot_guarantee_info:
                guarantee_value_raw_ecdf = ecdf_plot_guarantee_info["value"]
                # ... (proportion_below_guarantee calculation as before) ...
                proportion_below_guarantee = 0.0
                if len(ecdf_x_values) > 0:
                    if guarantee_value_raw_ecdf < ecdf_x_values[0]: proportion_below_guarantee = 0.0
                    elif guarantee_value_raw_ecdf >= ecdf_x_values[-1]: proportion_below_guarantee = 1.0
                    else:
                        idx = np.searchsorted(ecdf_x_values, guarantee_value_raw_ecdf, side='right')
                        if idx > 0: proportion_below_guarantee = ecdf_y_values[idx-1] 

                annotation_text_ecdf = f"{ecdf_plot_guarantee_info['label_short']}: ${guarantee_value_raw_ecdf:,.0f}<br>({proportion_below_guarantee*100:.1f}% at/below)"
                # ... (add vline for ecdf_plot_guarantee_info to fig_ecdf as before) ...
                current_plot_ecdf_x_min = ecdf_x_values.min()
                current_plot_ecdf_x_max = ecdf_x_values.max()
                plot_display_ecdf_max_x = current_plot_ecdf_x_max + 0.05 * (current_plot_ecdf_x_max - current_plot_ecdf_x_min if current_plot_ecdf_x_max > current_plot_ecdf_x_min else 1)
                plot_display_ecdf_min_x = current_plot_ecdf_x_min - 0.05 * (current_plot_ecdf_x_max - current_plot_ecdf_x_min if current_plot_ecdf_x_max > current_plot_ecdf_x_min else 1)
                if plot_display_ecdf_min_x <= guarantee_value_raw_ecdf <= plot_display_ecdf_max_x :
                    fig_ecdf.add_vline(x=guarantee_value_raw_ecdf,line_color="seagreen", line_dash="dashdot",annotation_text=annotation_text_ecdf,annotation_position="bottom left")

            fig_ecdf.update_layout(xaxis_tickformat='$,.0f', xaxis_tickangle=90)
            st.plotly_chart(fig_ecdf, key="ecdf_plot_exposure_stats")
            
            if not pd.isna(median_val_ecdf): st.success(f"Median {plot_title_status} Net Prize Money: ${median_val_ecdf:,.0f}")
            else: st.info("Median could not be calculated.")
        else:
            st.info("Not enough data to generate ECDF plot.")
            
        # --- NEW: Exposure Stats for Selected Rank Range ---
        if use_rank_filter:
            # The relevant guarantee for the *active filter* is plot_display_guarantee_info (since it was derived using sgl_rank_range_applied)
            active_filter_guarantee_info = plot_display_guarantee_info 

            if active_filter_guarantee_info:
                target_guarantee_val_for_calc = active_filter_guarantee_info["value"]
                
                # Conditions for exposure calculation on the 'filtered' DataFrame
                snumtrn_cond_exposure = pd.Series(True, index=filtered.index)
                if 'snumtrn' in filtered.columns: snumtrn_cond_exposure = (filtered['snumtrn'] > 14)
                
                carprz_cond_exposure = pd.Series(True, index=filtered.index)
                if 'carprz' in filtered.columns: carprz_cond_exposure = (filtered['carprz'] < 15_000_000)

                earnings_cond_exposure = (filtered[earnings_column] < target_guarantee_val_for_calc)
                final_mask_range_exposure = snumtrn_cond_exposure & carprz_cond_exposure & earnings_cond_exposure
                
                players_in_range_df = filtered[final_mask_range_exposure] # Already filtered by rank range
                num_players_in_range_exposed = len(players_in_range_df)
                
                total_exposure_val_in_range = 0
                if num_players_in_range_exposed > 0:
                    total_exposure_val_in_range = (target_guarantee_val_for_calc - players_in_range_df[earnings_column]).sum()

                exp_num_html = f"""<div style="background-color: #ffe0b3; color: #804000; border: 1px solid #ffcc80; border-radius: 0.25rem; padding: 0.5rem 1rem; font-weight: bold;">For Ranks {sgl_rank_range_applied[0]}-{sgl_rank_range_applied[1]} (vs G'tee ${target_guarantee_val_for_calc:,.0f}):<br># Players Exposed: {num_players_in_range_exposed}</div>"""
                exp_total_html = f"""<div style="background-color: #ffe0b3; color: #804000; border: 1px solid #ffcc80; border-radius: 0.25rem; padding: 0.5rem 1rem; font-weight: bold;">For Ranks {sgl_rank_range_applied[0]}-{sgl_rank_range_applied[1]} (vs G'tee ${target_guarantee_val_for_calc:,.0f}):<br>Total Exposure: ${total_exposure_val_in_range:,.0f}</div>"""
                st.markdown(f"""<div style='display: flex; flex-wrap: wrap; gap: 0.75rem; margin-top: 0.5rem; margin-bottom: 1rem;'>{exp_num_html}{exp_total_html}</div>""", unsafe_allow_html=True)
            elif use_rank_filter: # use_rank_filter is true, but no specific guarantee band matched the custom range well enough
                    st.markdown(f"""<div style='margin-top: 0.5rem; margin-bottom: 1rem; font-style: italic;'>No specific ATP guarantee band strongly aligns with the selected rank range {sgl_rank_range_applied[0]}-{sgl_rank_range_applied[1]} for detailed exposure stats display here.</div>""", unsafe_allow_html=True)


    # --- Exposure Comparison Section (remains largely the same) ---
    st.markdown("---")
    st.header("Exposure Comparison") 
    st.markdown("---")
    
    actual_exposure_years_to_calc = [] 
    if not use_adjusted_earnings: 
        actual_exposure_years_to_calc = selected_years
    else: 
        if selected_years:
            actual_exposure_years_to_calc = [by - 1 for by in selected_years]
            available_actual_rank_years_in_df = df['Year'].dropna().unique()
            actual_exposure_years_to_calc = [y for y in actual_exposure_years_to_calc if y in available_actual_rank_years_in_df]

    snumtrn_exists_globally = 'snumtrn' in df.columns
    carprz_exists_globally = 'carprz' in df.columns

    if not actual_exposure_years_to_calc:
        st.info("No Rank Years selected or available for Actual Exposure calculation based on current filters.")
    else:
        st.subheader(f"Actual Exposure Analysis (for Rank Year(s): {', '.join(map(str, sorted(list(set(actual_exposure_years_to_calc)))))} using '{'Net Prize Money (Actual)'}')")
        
        if not snumtrn_exists_globally: st.warning("Column 'snumtrn' not found. For Actual Exposure, players cannot meet 'games played > 14' condition.")
        if not carprz_exists_globally: st.warning("Column 'carprz' not found. For Actual Exposure, players cannot meet 'career prize < $15M' condition.")

        results_actual_exposure = [] 
        for year_val in sorted(list(set(actual_exposure_years_to_calc))):
            df_year_actual = df[df['Year'] == year_val].copy()
            count_actual0, exposure_actual0 = 0, 0 
            count_actual1, exposure_actual1 = 0, 0 
            count_actual2, exposure_actual2 = 0, 0 
            comment_for_year = None

            if df_year_actual.empty: comment_for_year = "No data for this year"
            else:
                base_mask_actual0 = (df_year_actual['sglrank'].between(51, 100) & (df_year_actual['Net Prize Money (Actual)'] < guarantee_51_100))
                base_mask_actual1 = (df_year_actual['sglrank'].between(101, 175) & (df_year_actual['Net Prize Money (Actual)'] < guarantee_101_175))
                base_mask_actual2 = (df_year_actual['sglrank'].between(176, 250) & (df_year_actual['Net Prize Money (Actual)'] < guarantee_176_250))

                snumtrn_filter_actual = pd.Series(True, index=df_year_actual.index)
                if snumtrn_exists_globally and 'snumtrn' in df_year_actual.columns: snumtrn_filter_actual = (df_year_actual['snumtrn'] > 14)
                
                carprz_filter_actual = pd.Series(True, index=df_year_actual.index)
                if carprz_exists_globally and 'carprz' in df_year_actual.columns: carprz_filter_actual = (df_year_actual['carprz'] < 15_000_000)

                mask_actual0 = base_mask_actual0 & snumtrn_filter_actual & carprz_filter_actual
                count_actual0 = mask_actual0.sum()
                if count_actual0 > 0: exposure_actual0 = (guarantee_51_100 - df_year_actual.loc[mask_actual0, 'Net Prize Money (Actual)']).sum()

                mask_actual1 = base_mask_actual1 & snumtrn_filter_actual & carprz_filter_actual
                count_actual1 = mask_actual1.sum()
                if count_actual1 > 0: exposure_actual1 = (guarantee_101_175 - df_year_actual.loc[mask_actual1, 'Net Prize Money (Actual)']).sum()

                mask_actual2 = base_mask_actual2 & snumtrn_filter_actual & carprz_filter_actual
                count_actual2 = mask_actual2.sum()
                if count_actual2 > 0: exposure_actual2 = (guarantee_176_250 - df_year_actual.loc[mask_actual2, 'Net Prize Money (Actual)']).sum()
            
            results_actual_exposure.append({
                "Year": year_val, "Count_51_100": count_actual0, "Exposure_51_100": exposure_actual0,
                "Count_101_175": count_actual1, "Exposure_101_175": exposure_actual1,
                "Count_176_250": count_actual2, "Exposure_176_250": exposure_actual2, "comment": comment_for_year
            })

        if results_actual_exposure:
            actual_exposure_df = pd.DataFrame(results_actual_exposure)
            actual_exposure_df = actual_exposure_df.sort_values(by='Year')
            plot_df_exposure = actual_exposure_df[actual_exposure_df['comment'].isna()].copy() # Renamed for clarity

            count_trace_names_exp = { # Renamed for clarity
                'Count_51_100': f'Ranks 51-100 (Target < ${guarantee_51_100:,.0f})',
                'Count_101_175': f'Ranks 101-175 (Target < ${guarantee_101_175:,.0f})',
                'Count_176_250': f'Ranks 176-250 (Target < ${guarantee_176_250:,.0f})'
            }
            amount_trace_names_exp = { # Renamed for clarity
                'Exposure_51_100': f'Ranks 51-100 (Target < ${guarantee_51_100:,.0f})',
                'Exposure_101_175': f'Ranks 101-175 (Target < ${guarantee_101_175:,.0f})',
                'Exposure_176_250': f'Ranks 176-250 (Target < ${guarantee_176_250:,.0f})'
            }

            if not plot_df_exposure.empty and plot_df_exposure['Year'].nunique() > 0 : 
                st.markdown("---")
                st.subheader("Trend of Actual Exposure Players by Year")
                fig_count_trend_exp = px.line(plot_df_exposure, x='Year', y=['Count_51_100', 'Count_101_175', 'Count_176_250'], labels={'value': 'Number of Players', 'Year': 'Rank Year'}, markers=True, title="Number of Players with Actual Exposure by Rank Year")
                fig_count_trend_exp.for_each_trace(lambda t: t.update(name=count_trace_names_exp.get(t.name, t.name)))
                fig_count_trend_exp.update_layout(legend_title_text='Player Groups')
                st.plotly_chart(fig_count_trend_exp)

                st.subheader("Trend of Actual Exposure Amount by Year")
                fig_amount_trend_exp = px.line(plot_df_exposure, x='Year', y=['Exposure_51_100', 'Exposure_101_175', 'Exposure_176_250'], labels={'value': 'Total Exposure Amount ($)', 'Year': 'Rank Year'}, markers=True, title="Actual Exposure Amount by Rank Year")
                fig_amount_trend_exp.for_each_trace(lambda t: t.update(name=amount_trace_names_exp.get(t.name, t.name)))
                fig_amount_trend_exp.update_layout(legend_title_text='Exposure Amounts', yaxis_tickformat='$,.0f')
                st.plotly_chart(fig_amount_trend_exp)
            elif not actual_exposure_df.empty : st.info("Not enough yearly data points to plot trends for actual exposure.")

            st.markdown("---")
            st.subheader("Detailed Actual Exposure by Year:")
            for _, row_data in actual_exposure_df.iterrows():
                st.markdown(f"**Rank Year: {int(row_data['Year'])}**")
                if row_data["comment"]:
                    st.write(row_data["comment"])
                    st.markdown("---")
                    continue
                col0_act, col1_act, col2_act = st.columns(3)
                col0_act.metric(label=f"Ranks 51–100: # below ${guarantee_51_100:,.0f} (Actual)", value=int(row_data['Count_51_100']), delta=f"${row_data['Exposure_51_100']:,.0f} total actual exposure", delta_color="off")
                col1_act.metric(label=f"Ranks 101–175: # below ${guarantee_101_175:,.0f} (Actual)", value=int(row_data['Count_101_175']), delta=f"${row_data['Exposure_101_175']:,.0f} total actual exposure", delta_color="off")
                col2_act.metric(label=f"Ranks 176–250: # below ${guarantee_176_250:,.0f} (Actual)", value=int(row_data['Count_176_250']), delta=f"${row_data['Exposure_176_250']:,.0f} total actual exposure", delta_color="off")
                st.markdown("---")
        else: st.info("No data processed for Actual Exposure calculation.")
else:
    st.warning("Please upload the earnings data CSV file to proceed.")