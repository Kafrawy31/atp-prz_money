import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde # iqr is not used, can be removed if not needed elsewhere
import plotly.express as px
import plotly.graph_objects as go

st.title("ATP Player Earnings Analysis")

# --- File uploader for CSV or Excel file ---
uploaded_file = st.file_uploader(
    "Upload earnings data (CSV, or Excel with a sheet named 'Ranking'):",
    type=["csv", "xlsx", "xls"]  # Accept all three types
)

if uploaded_file is not None:
    df = None  # Initialize df to None
    file_name = uploaded_file.name.lower() # Get lowercase filename for extension checking

    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, parse_dates=["rankdate"])
            st.success(f"Successfully loaded CSV file: {uploaded_file.name}")
        elif file_name.endswith(('.xlsx', '.xls')):
            try:
                df = pd.read_excel(
                    uploaded_file,
                    sheet_name="Ranking",
                    parse_dates=["rankdate"]
                )
                st.success(f"Successfully loaded Excel file: {uploaded_file.name} (from sheet: 'Ranking')")
            except ValueError as e:
                # More specific check for sheet not found errors
                if "Worksheet named 'Ranking' not found" in str(e) or \
                   "No sheet named <'Ranking'>" in str(e) or \
                   "No sheet named 'Ranking'" in str(e): # Check for common pandas sheet not found messages
                    st.error(f"Error in Excel file '{uploaded_file.name}': Could not find a sheet named 'Ranking'. Please ensure this sheet exists.")
                    st.stop()
                else:
                    # Other ValueErrors during Excel parsing (e.g., corrupted file)
                    st.error(f"Error reading Excel file '{uploaded_file.name}': {e}. The file might be corrupted or not a valid Excel format.")
                    st.stop()
        else:
            # This case should ideally not be reached if `type` in file_uploader works as expected
            st.error(f"Unsupported file type: {uploaded_file.name}. Please upload a CSV or Excel file (.xlsx, .xls).")
            st.stop()

    except pd.errors.ParserError as pe: # Catch CSV parsing errors specifically
        st.error(f"Error parsing CSV file '{uploaded_file.name}': {pe}. Please ensure it's a valid CSV.")
        st.stop()
    except Exception as e:
        # Catch any other unexpected errors during file reading or initial processing
        st.error(f"An unexpected error occurred while processing the file '{uploaded_file.name}': {e}")
        st.stop()

    if df is None: # Final safeguard, though specific errors should have stopped execution
        st.error("Failed to load data from the uploaded file. Please check the file and try again.")
        st.stop()

    # --- Continue with your existing data processing from here ---
    # Ensure 'rankdate' was parsed correctly; if not, it might cause errors below
    if 'rankdate' not in df.columns or pd.api.types.is_datetime64_any_dtype(df['rankdate']) is False:
        # Check if 'rankdate' exists and is actually datetime
        # Attempt to convert if it's not, or warn if it's missing critical data
        if 'rankdate' in df.columns:
            try:
                df['rankdate'] = pd.to_datetime(df['rankdate'])
                if pd.api.types.is_datetime64_any_dtype(df['rankdate']) is False: # Check again
                    st.warning("Could not convert 'rankdate' column to datetime. Date-related features may fail.")
            except Exception as date_conv_e:
                st.warning(f"Failed to convert 'rankdate' to datetime automatically: {date_conv_e}. Please check the 'rankdate' column format.")
        else:
            st.error("Critical column 'rankdate' not found in the uploaded data. Application cannot proceed.")
            st.stop()


    df['Year'] = df['rankdate'].dt.year + 1
    df['Baseline Year'] = df['rankdate'].dt.year + 1

    sglrank_numeric = pd.to_numeric(df['sglrank'], errors='coerce')
    sgl_rank_min_df_default, sgl_rank_max_df_default = (1, 250)  # Sensible defaults
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

    if 'Signed Policy' in df.columns:
        df['Signed Policy_Original'] = df['Signed Policy']
    st.sidebar.title("Filters")

    # --- Dynamic Guarantee Value Inputs (in Thousands of Dollars) ---

    use_adjusted_earnings = st.sidebar.checkbox("Use adjusted earnings (2025)", value=False)
    year_column = 'Baseline Year'
    
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
    # Add 'Signed Policy' to essential_cols check if it becomes truly essential for core functionality
    # For now, we'll warn if it's missing when the filter is used.
    missing_essential_cols = [col for col in essential_cols if col not in df.columns]
    if missing_essential_cols:
        st.error(f"Essential column(s) not found: {', '.join(missing_essential_cols)}.")
        st.stop()
    
    exposure_condition_cols = ['Events Played', 'carprz'] # 'Signed Policy' will be checked separately
    for col in exposure_condition_cols:
        if col not in df.columns:
            st.warning(f"Column '{col}' not found. Exposure calculations might be affected.")

    # --- NEW: Signed Players Filter ---
    use_signed_players_filter = st.sidebar.checkbox("Use signed players only", value=False)
    signed_policy_col_exists = 'Signed Policy' in df.columns

    if use_signed_players_filter and not signed_policy_col_exists:
        st.sidebar.warning("Column 'Signed Policy' not found. Cannot apply 'Use signed players only' filter.")
        use_signed_players_filter = False # Disable filter if column is missing

        # --- Signature Projection Controls ---
    st.sidebar.markdown("### Signature Projections")
    project_by_player = st.sidebar.checkbox("Project Signatures by Player (plyrnum)", value=False)
    project_by_rank = st.sidebar.checkbox("Project Signatures by Rank (sglrank)", value=False)

    if project_by_player and project_by_rank:
        st.sidebar.warning("Please select only one projection method (player or rank).")
    else:
        if 'Signed Policy' in df.columns and 'Baseline Year' in df.columns:
            max_year = df['Baseline Year'].max()
            prev_year = max_year - 1

            if project_by_player or project_by_rank:
                # Wipe 2025 signed policy clean
                df.loc[df['Baseline Year'] == max_year, 'Signed Policy'] = None

                if project_by_player:
                    signed_players = df[(df['Baseline Year'] == prev_year) & (df['Signed Policy'] == 'P')]['plyrnum'].unique()
                    df.loc[(df['Baseline Year'] == max_year) & (df['plyrnum'].isin(signed_players)), 'Signed Policy'] = 'P'

                elif project_by_rank:
                    signed_ranks = df[(df['Baseline Year'] == prev_year) & (df['Signed Policy'] == 'P')]['sglrank'].unique()
                    df.loc[(df['Baseline Year'] == max_year) & (df['sglrank'].isin(signed_ranks)), 'Signed Policy'] = 'P'

            else:
                # If no projection is selected, restore original 'P' values
                df.loc[df['Baseline Year'] == max_year, 'Signed Policy'] = df.loc[df['Baseline Year'] == max_year, 'Signed Policy_Original']
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
    guarantee_1_100_k = st.sidebar.number_input("Ranks 1-100 Guarantee ($k)", value=300, min_value=0, step=10, format="%d", key="g1_100k_usd")
    guarantee_101_175_k = st.sidebar.number_input("Ranks 101-175 Guarantee ($k)", value=200, min_value=0, step=10, format="%d", key="g101_175k_usd")
    guarantee_176_250_k = st.sidebar.number_input("Ranks 176-250 Guarantee ($k)", value=100, min_value=0, step=10, format="%d", key="g176_250k_usd")

    st.sidebar.subheader("Guarantee Multipliers")
    multiplier_1_100 = st.sidebar.number_input("Multiplier for Ranks 1-100", min_value=0.01, value=1.0, step=0.05, format="%.2f", key="mult1_100")
    multiplier_101_175 = st.sidebar.number_input("Multiplier for Ranks 101-175", min_value=0.01, value=1.0, step=0.05, format="%.2f", key="mult101_175")
    multiplier_176_250 = st.sidebar.number_input("Multiplier for Ranks 176-250", min_value=0.01, value=1.0, step=0.05, format="%.2f", key="mult176_250")

    base_guarantee_1_100_val = guarantee_1_100_k * 1000
    base_guarantee_101_175_val = guarantee_101_175_k * 1000
    base_guarantee_176_250_val = guarantee_176_250_k * 1000
    
    # Apply multipliers to get final guarantee values (These were not being multiplied in the previous version)
    guarantee_1_100 = base_guarantee_1_100_val
    guarantee_101_175 = base_guarantee_101_175_val
    guarantee_176_250 = base_guarantee_176_250_val

    guarantee_map_dynamic = [
        {"min_r": 1,   "max_r": 100, "value": guarantee_1_100, "label_short": "G'tee (1-100)", "label_full": f"Guarantee (Ranks 1-100: ${guarantee_1_100:,.0f})"},
        {"min_r": 101, "max_r": 175, "value": guarantee_101_175, "label_short": "G'tee (101-175)", "label_full": f"Guarantee (Ranks 101-175: ${guarantee_101_175:,.0f})"},
        {"min_r": 176, "max_r": 250, "value": guarantee_176_250, "label_short": "G'tee (176-250)", "label_full": f"Guarantee (Ranks 176-250: ${guarantee_176_250:,.0f})"},
    ]
    
    def on_preset_change():
        preset_name = st.session_state.rank_preset_radio 
        if preset_name != "Custom":
            min_r, max_r = rank_presets_options[preset_name]
            st.session_state.sgl_rank_min_val = min_r
            st.session_state.sgl_rank_max_val = max_r
            st.session_state.use_rank_filter_val = True
    
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
        key='rank_preset_radio', 
        on_change=on_preset_change,
        index=list(rank_presets_options.keys()).index(current_filter_is_preset) 
    )

    st.session_state.use_rank_filter_val = st.sidebar.checkbox(
        "Filter by Rank Range (enable for presets or custom)",
        key='use_rank_filter_val_cb', 
        value=st.session_state.use_rank_filter_val
    )
    use_rank_filter = st.session_state.use_rank_filter_val 

    sgl_rank_range_applied = (sgl_rank_min_df_default, sgl_rank_max_df_default) 

    if use_rank_filter:
        st.sidebar.write("Define Rank Range:")
        new_min_rank = st.sidebar.number_input(
            "Min SGL Rank",
            min_value=1, max_value=sgl_rank_max_df_default + 500, 
            value=st.session_state.sgl_rank_min_val,
            key='sgl_rank_min_input_widget' 
        )
        new_max_rank = st.sidebar.number_input(
            "Max SGL Rank",
            min_value=1, max_value=sgl_rank_max_df_default + 500,
            value=st.session_state.sgl_rank_max_val,
            key='sgl_rank_max_input_widget' 
        )
        if new_min_rank != st.session_state.sgl_rank_min_val or new_max_rank != st.session_state.sgl_rank_max_val:
            st.session_state.sgl_rank_min_val = new_min_rank
            st.session_state.sgl_rank_max_val = new_max_rank
            st.session_state.rank_preset_key = "Custom" 

        if st.session_state.sgl_rank_min_val > st.session_state.sgl_rank_max_val:
            st.session_state.sgl_rank_max_val = st.session_state.sgl_rank_min_val
        
        sgl_rank_range_applied = (st.session_state.sgl_rank_min_val, st.session_state.sgl_rank_max_val)
    
    # --- Other Filters ---
    use_snumtrn_filter = st.sidebar.checkbox("Use Tournament Filter (Events Played)",value = True)
    snumtrn_range = None
    if use_snumtrn_filter and 'Events Played' in df.columns and not df['Events Played'].dropna().empty:
        snumtrn_min, snumtrn_max = int(df['Events Played'].dropna().min()), int(df['Events Played'].dropna().max())
        snumtrn_range = st.sidebar.slider(
            "Select number of tournaments Range",
            min_value=snumtrn_min, max_value=snumtrn_max,
            value=(15, snumtrn_max)
        )
    elif use_snumtrn_filter:
        st.sidebar.warning("'Events Played' column not found or empty; cannot apply tournament filter.")

    use_carprz_filter = st.sidebar.checkbox("Use career prize Filter (carprz)", value = True)
    carprz_range = None 
    if use_carprz_filter and 'carprz' in df.columns and not df['carprz'].dropna().empty:
        carprz_min_val, carprz_max_val = int(df['carprz'].dropna().min()), int(df['carprz'].dropna().max())
        st.sidebar.write("Enter career prize Range ($):") 
        carprz_min_input = st.sidebar.number_input(
            "Min career prize ($)",
            value=carprz_min_val, min_value=carprz_min_val, max_value=carprz_max_val, format="%d"
        )
        carprz_max_input = st.sidebar.number_input(
            "Max career prize ($)",
            value=min(carprz_max_val, 15_000_000), 
            min_value=carprz_min_val, max_value=carprz_max_val, format="%d"
        )
        carprz_range = (carprz_min_input, carprz_max_input)
    elif use_carprz_filter:
         st.sidebar.warning("'carprz' column not found or empty; cannot apply career prize filter.")

    use_prize_money_filter = st.sidebar.checkbox("Use Prize Money Filter")
    prize_range = None
    if use_prize_money_filter and earnings_column in df.columns:
        if not df[earnings_column].dropna().empty:
            prize_min_val, prize_max_val = int(df[earnings_column].dropna().min()), int(df[earnings_column].dropna().max()) 
            st.sidebar.write("Enter Prize Money Range ($):") 
            prize_min_input = st.sidebar.number_input("Min Prize Money ($)", value=prize_min_val, min_value=prize_min_val, max_value=prize_max_val, format="%d")
            prize_max_input = st.sidebar.number_input("Max Prize Money ($)", value=prize_max_val, min_value=prize_min_val, max_value=prize_max_val, format="%d")
            prize_range = (prize_min_input, prize_max_input)
        else:
            st.sidebar.warning(f"No data in '{earnings_column}' for prize money filter.")
            use_prize_money_filter = False 
    elif use_prize_money_filter: 
        st.sidebar.warning(f"'{earnings_column}' column not found for prize money filter.")
        use_prize_money_filter = False 


    # --- Expected Exposure Summary (uses full guarantee values) ---
    st.subheader("Expected Exposure Summary (for Baseline Year 2025, using Adjusted Earnings)")
    st.markdown(f"Players must have >14 tournaments played and < $15M in total career earnings. Note, guarantee thresholds are user configurable (including multipliers). {'Only signed players included.' if use_signed_players_filter else ''}")
    df_baseline_2025 = df[df['Baseline Year'] == 2025].copy()
    
    count0_expected, exposure0_expected = 0, 0 
    count1_expected, exposure1_expected = 0, 0 
    count2_expected, exposure2_expected = 0, 0 

    if not df_baseline_2025.empty:
        base_condition0_expected = (df_baseline_2025['sglrank'].between(1, 100) & (df_baseline_2025['Net Prize Money (2025 Adjusted)'] < guarantee_1_100))
        base_condition1_expected = (df_baseline_2025['sglrank'].between(101, 175) & (df_baseline_2025['Net Prize Money (2025 Adjusted)'] < guarantee_101_175))
        base_condition2_expected = (df_baseline_2025['sglrank'].between(176, 250) & (df_baseline_2025['Net Prize Money (2025 Adjusted)'] < guarantee_176_250))

        snumtrn_filter_expected = pd.Series(True, index=df_baseline_2025.index) 
        if snumtrn_range and 'Events Played' in df_baseline_2025.columns:
            df_baseline_2025 = df_baseline_2025[
                df_baseline_2025['Events Played'].between(snumtrn_range[0], snumtrn_range[1])
            ]
        
        carprz_filter_expected = pd.Series(True, index=df_baseline_2025.index) 
        if 'carprz' in df_baseline_2025.columns:
            carprz_filter_expected = (df_baseline_2025['carprz'] < 15_000_000) 

        signed_player_filter_expected = pd.Series(True, index=df_baseline_2025.index)
        if use_signed_players_filter and signed_policy_col_exists:
            signed_player_filter_expected = df_baseline_2025['Signed Policy'].astype(str).str.contains('P', na=False)
        
        mask0_expected = base_condition0_expected & snumtrn_filter_expected & carprz_filter_expected & signed_player_filter_expected
        count0_expected = mask0_expected.sum()
        if count0_expected > 0: 
            exposure0_expected = (guarantee_1_100 - df_baseline_2025.loc[mask0_expected, 'Net Prize Money (2025 Adjusted)']).sum() * multiplier_1_100

        mask1_expected = base_condition1_expected & snumtrn_filter_expected & carprz_filter_expected & signed_player_filter_expected
        count1_expected = mask1_expected.sum()
        if count1_expected > 0: 
            exposure1_expected = (guarantee_101_175 - df_baseline_2025.loc[mask1_expected, 'Net Prize Money (2025 Adjusted)']).sum() * multiplier_101_175

        mask2_expected = base_condition2_expected & snumtrn_filter_expected & carprz_filter_expected & signed_player_filter_expected
        count2_expected = mask2_expected.sum()
        if count2_expected > 0: 
            exposure2_expected = (guarantee_176_250 - df_baseline_2025.loc[mask2_expected, 'Net Prize Money (2025 Adjusted)']).sum() * multiplier_176_250
    
    c0_exp, c1_exp, c2_exp = st.columns(3)
    c0_exp.metric(f"Ranks 1–100: # below ${guarantee_1_100:,.0f}", count0_expected, f"${exposure0_expected:,.0f} total expected exposure")
    c1_exp.metric(f"Ranks 101–175: # below ${guarantee_101_175:,.0f}", count1_expected, f"${exposure1_expected:,.0f} total expected exposure")
    c2_exp.metric(f"Ranks 176–250: # below ${guarantee_176_250:,.0f}", count2_expected, f"${exposure2_expected:,.0f} total expected exposure")

    # Compute total expected players and exposure
    total_expected_count = count0_expected + count1_expected + count2_expected
    total_expected_exposure = exposure0_expected + exposure1_expected + exposure2_expected

    # Display total
    st.markdown(
        f"""
        <div style="background-color: #0f0f0f; border: 1px solid #cce5ff; border-radius: 8px;
                    padding: 10px; margin-top: 1rem;">
            <strong>Total Expected (All Ranks):</strong><br>
            <span style="font-size: 0.9rem;">
                Players Below Threshold: <strong>{total_expected_count}</strong><br>
                Total Expected Exposure: <strong>${total_expected_exposure:,.0f}</strong>
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # --- Data Filtering ---
    # Start with a fresh copy for the main filtered section
    filtered_display = df.copy() 

    if year_column in df.columns and selected_years:
        filtered_display = filtered_display[filtered_display[year_column].isin(selected_years)]
    elif not selected_years and year_column in df.columns and df[year_column].nunique() > 0 :
        pass 
    elif not year_column in df.columns or (year_column in df.columns and df[year_column].nunique() == 0):
        st.warning(f"'{year_column}' not found or has no data. Cannot filter by year for main display.")
        # filtered_display = pd.DataFrame(columns=df.columns) 

    if use_signed_players_filter and signed_policy_col_exists and not filtered_display.empty :
         filtered_display = filtered_display[filtered_display['Signed Policy'].astype(str).str.contains('P', na=False)]
    elif use_signed_players_filter and not signed_policy_col_exists: # Should have been caught by sidebar warning
        st.warning("Cannot apply signed player filter to main display as 'Signed Policy' column is missing.")


    if use_rank_filter: 
        if 'sglrank' in filtered_display.columns:
            filtered_display = filtered_display[(filtered_display['sglrank'] >= sgl_rank_range_applied[0]) & (filtered_display['sglrank'] <= sgl_rank_range_applied[1])]
    
    if use_snumtrn_filter and snumtrn_range and 'Events Played' in filtered_display.columns and not filtered_display.empty:
        filtered_display = filtered_display[(filtered_display['Events Played'] >= snumtrn_range[0]) & (filtered_display['Events Played'] <= snumtrn_range[1])]
    if use_carprz_filter and carprz_range and 'carprz' in filtered_display.columns and not filtered_display.empty:
        filtered_display = filtered_display[(filtered_display['carprz'] >= carprz_range[0]) & (filtered_display['carprz'] <= carprz_range[1])]
    if use_prize_money_filter and prize_range and earnings_column in filtered_display.columns and not filtered_display.empty:
         if not filtered_display[earnings_column].dropna().empty : 
            filtered_display = filtered_display[(filtered_display[earnings_column] >= prize_range[0]) & (filtered_display[earnings_column] <= prize_range[1])]

    earnings = pd.Series(dtype=float)
    if earnings_column in filtered_display.columns and not filtered_display[earnings_column].dropna().empty:
        earnings = filtered_display[earnings_column].dropna().sort_values().reset_index(drop=True)

    # --- Main Title ---
    title_year_prefix = "Baseline Year(s):"
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
    signed_filter_title_addon = " (Signed Players Only)" if use_signed_players_filter and signed_policy_col_exists else ""

    main_title_text = (f"**{title_year_prefix}** {years_display_string_main} • **{rank_display_string_main}** • **{title_earnings_suffix}**{signed_filter_title_addon}")
    st.markdown(main_title_text)


    # --- Helper function for more lenient guarantee matching ---
    def get_relevant_guarantee_info_for_display(user_min_r, user_max_r, is_filter_active, current_filtered_df_helper, guarantee_map): # Renamed current_filtered_df to current_filtered_df_helper
        if is_filter_active:
            for g_info in guarantee_map:
                if user_min_r >= g_info["min_r"] and user_max_r <= g_info["max_r"]:
                    return g_info 
            user_mid_r = (user_min_r + user_max_r) / 2
            for g_info in guarantee_map:
                overlap_min = max(user_min_r, g_info["min_r"])
                overlap_max = min(user_max_r, g_info["max_r"])
                if overlap_min <= overlap_max: 
                    if g_info["min_r"] <= user_mid_r <= g_info["max_r"]:
                        return g_info 
            largest_overlap_amount = 0
            most_overlapped_g_info = None
            for g_info in guarantee_map:
                overlap_start = max(user_min_r, g_info["min_r"])
                overlap_end = min(user_max_r, g_info["max_r"])
                overlap_length = overlap_end - overlap_start
                if overlap_length > 0: 
                    user_range_width = user_max_r - user_min_r + 1
                    g_band_width = g_info["max_r"] - g_info["min_r"] + 1
                    if overlap_length > largest_overlap_amount and (overlap_length / user_range_width > 0.4 or overlap_length / g_band_width > 0.4) : 
                        largest_overlap_amount = overlap_length
                        most_overlapped_g_info = g_info
            if most_overlapped_g_info:
                return most_overlapped_g_info
            return None 
        else:
            if not current_filtered_df_helper.empty and 'sglrank' in current_filtered_df_helper.columns and current_filtered_df_helper['sglrank'].nunique() > 0:
                data_median_rank_for_plot = current_filtered_df_helper['sglrank'].median()
                for g_info_fallback in guarantee_map:
                    if g_info_fallback["min_r"] <= data_median_rank_for_plot <= g_info_fallback["max_r"]:
                        data_min_rank_for_plot = current_filtered_df_helper['sglrank'].min()
                        data_max_rank_for_plot = current_filtered_df_helper['sglrank'].max()
                        if (data_max_rank_for_plot - data_min_rank_for_plot) < ((g_info_fallback["max_r"] - g_info_fallback["min_r"]) + 75): 
                            return g_info_fallback
            return None

    if earnings.empty: # earnings is derived from filtered_display
        st.warning("No data available for the selected filter combination (including signed player filter if active).")
    else:
        plot_title_status = "Adjusted" if use_adjusted_earnings else "Actual"
        years_str_for_plot_title = "Selected Year(s)" 
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


        # --- Histogram ---
        st.subheader("Prize Money Distribution")
        if earnings.empty:
            st.warning("No earnings data to plot for histogram.")
        else:
            # --- Filter earnings by guarantee threshold if rank preset selected ---
            earnings_filtered = earnings.copy()

            preset_range = rank_presets_options.get(st.session_state.rank_preset_radio)
            if preset_range:
                min_r, max_r = preset_range
                guarantee_value = None
                if max_r <= 100:
                    guarantee_value = guarantee_1_100
                elif max_r <= 175:
                    guarantee_value = guarantee_101_175
                else:
                    guarantee_value = guarantee_176_250
                
                if guarantee_value is not None and 'sglrank' in filtered_display.columns:
                    earnings_filtered = filtered_display[
                        (filtered_display['sglrank'].between(min_r, max_r)) &
                        (filtered_display[earnings_column] < guarantee_value)
                    ][earnings_column].dropna().sort_values().reset_index(drop=True)

            else:
                earnings_filtered = earnings.copy()  # Default to all earnings

            # If filtered earnings are empty, show message
            if earnings_filtered.empty:
                st.warning("No players below the guarantee threshold for the selected rank bucket.")
            else:
                actual_min_earning = earnings_filtered.min()
                actual_max_earning = earnings_filtered.max()
                hist_range_x_min = 0
                if actual_min_earning < 0: hist_range_x_min = actual_min_earning
                hist_range_x_max = actual_max_earning
                if actual_max_earning <= hist_range_x_min:
                    hist_range_x_max = hist_range_x_min + (100000 if hist_range_x_min == 0 else abs(hist_range_x_min * 0.5) + 1)

                num_bins = 30
                if hist_range_x_min < hist_range_x_max:
                    counts_hist, bin_edges_hist = np.histogram(earnings_filtered, bins=num_bins, range=(hist_range_x_min, hist_range_x_max))
                else:
                    counts_hist, bin_edges_hist = np.histogram(earnings_filtered, bins=num_bins)

                bin_labels = [f"${bin_edges_hist[i]:,.0f} - ${bin_edges_hist[i+1]:,.0f}" for i in range(len(counts_hist))]
                hist_plot_data = pd.DataFrame({
                    'Bin_Range_Label': bin_labels,
                    'Player_Count': counts_hist,
                    'Bin_Lower_Edge': bin_edges_hist[:-1],
                    'Bin_Upper_Edge': bin_edges_hist[1:]
                })

                max_count_hist = max(counts_hist) if len(counts_hist) > 0 else 0
                padded_max_hist = max_count_hist * 1.8 if max_count_hist > 0 else 10

                fig_hist = px.bar(
                    hist_plot_data,
                    x='Bin_Range_Label',
                    y='Player_Count',
                    text='Player_Count',
                    custom_data=['Bin_Lower_Edge', 'Bin_Upper_Edge']
                )
                fig_hist.update_traces(textposition='outside')
                fig_hist.update_layout(
                    title_text=f"Distribution of {plot_title_status} Prize Money for {years_str_for_plot_title}{signed_filter_title_addon}",
                    yaxis=dict(range=[0, padded_max_hist], title_text="Number of Players"),
                    xaxis=dict(title_text=f"{earnings_column} Bins", tickangle=-90, type='category'),
                    bargap=0.1
                )
                fig_hist.update_traces(hovertemplate=(
                    f"<b>{earnings_column} Range:</b> $%{{customdata[0]:,.0f}} - $%{{customdata[1]:,.0f}}<br>"
                    "<b>Number of Players:</b> %{y}<extra></extra>"
                ))
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
            num_players_below_lower_bound = (earnings_m < lower_bound_mad).sum()
            kde = gaussian_kde(earnings_m)
            min_earnings_m, max_earnings_m = earnings_m.min(), earnings_m.max()
            if min_earnings_m == max_earnings_m: 
                x_range_kde = np.linspace(min_earnings_m - 0.1 * abs(min_earnings_m) if min_earnings_m != 0 else -0.1, 
                                          max_earnings_m + 0.1 * abs(max_earnings_m) if max_earnings_m != 0 else 0.1, 200)
            else:
                x_range_kde = np.linspace(min_earnings_m, max_earnings_m, 200)
            y_kde = kde(x_range_kde)
            
            fig_kde = px.line(x=x_range_kde, y=y_kde, labels={'x': f'{earnings_column} (Millions $)', 'y': 'Density'})
            fig_kde.update_layout(title_text=f"Density of {plot_title_status} Prize Money ({years_str_for_plot_title}{signed_filter_title_addon}, Median & Scaled MAD)", xaxis_tickformat='$,.3f', xaxis_tickangle=90) 
            fig_kde.add_vline(x=median_val_m, line_color="blue", line_dash="dot", annotation_text=f"Median: ${median_val_m:,.3f}m")
            fig_kde.add_vline(x=lower_bound_mad, line_color="purple", line_dash="dash", annotation_text=f"Lower MAD Bound: ${lower_bound_mad:,.3f}m") 
            fig_kde.add_vline(x=upper_bound_mad, line_color="purple", line_dash="dash", annotation_text=f"Upper MAD Bound: ${upper_bound_mad:,.3f}m")
            
            plot_display_guarantee_info = get_relevant_guarantee_info_for_display(
                sgl_rank_range_applied[0], sgl_rank_range_applied[1], use_rank_filter, filtered_display, guarantee_map_dynamic # Pass filtered_display
            )
            
            if plot_display_guarantee_info:
                guarantee_val_m = plot_display_guarantee_info["value"] / 1_000_000 
                annotation_text_for_line = f"{plot_display_guarantee_info['label_short']}: ${guarantee_val_m:,.3f}m" 
                
                plot_x_min_kde, plot_x_max_kde = x_range_kde.min(), x_range_kde.max() 
                if (plot_x_min_kde - 0.05 * (plot_x_max_kde - plot_x_min_kde if plot_x_max_kde > plot_x_min_kde else 1)) <= guarantee_val_m <= \
                   (plot_x_max_kde + 0.05 * (plot_x_max_kde - plot_x_min_kde if plot_x_max_kde > plot_x_min_kde else 1)):
                    fig_kde.add_vline(x=guarantee_val_m, line_color="seagreen", line_dash="longdashdot", annotation_text=annotation_text_for_line, annotation_position="bottom right")

            fig_kde.update_traces(hovertemplate=f'{earnings_column} (Millions $): %{{x:.3f}}<br>Density: %{{y:.3f}}<extra></extra>')
            st.plotly_chart(fig_kde, key="kde_density_plot_exposure_stats")

            median_display_html = f"""<div style="background-color: #e0e0ff; color: #000080; border: 1px solid #b0b0e0; border-radius: 0.25rem; padding: 0.5rem 1rem; font-weight: bold; font-size: 0.75em;">Median: ${median_val_m * 1e6:,.0f}</div>"""
            upper_mad_display_html = f"""<div style="background-color: #f0e6ff; color: #4b0082; border: 1px solid #d8c0ff; border-radius: 0.25rem; padding: 0.5rem 1rem; font-weight: bold; font-size: 0.75em;">Upper Bound (Scaled MAD): ${upper_bound_mad * 1e6:,.0f}</div>"""
            lower_mad_display_html = f"""<div style="background-color: #f0e6ff; color: #4b0082; border: 1px solid #d8c0ff; border-radius: 0.25rem; padding: 0.5rem 1rem; font-weight: bold; font-size: 0.75em;">
                                    Lower Bound (Scaled MAD): ${lower_bound_mad * 1e6:,.0f}<br>
                                    Players below: {num_players_below_lower_bound}
                                    </div>"""

            general_guarantee_display_html = ""
            if plot_display_guarantee_info: 
                general_guarantee_text_for_box = f"{plot_display_guarantee_info['label_full']}" 
                general_guarantee_display_html = f"""<div style="background-color: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; border-radius: 0.25rem; padding: 0.5rem 1rem; font-weight: bold; font-size: 0.75em;">Relevant Guarantee: {general_guarantee_text_for_box}</div>"""

            current_display_elements = [median_display_html, upper_mad_display_html,lower_mad_display_html]
            if general_guarantee_display_html:
                current_display_elements.append(general_guarantee_display_html)
            
            st.markdown(f"""<div style='display: flex; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 1rem;'>{''.join(current_display_elements)}</div>""", unsafe_allow_html=True)

            st.info(f"Percentage of players (in current filtered data) within Median ± Scaled MAD: {(len(earnings_m[(earnings_m >= lower_bound_mad) & (earnings_m <= upper_bound_mad)]) / len(earnings_m) * 100 if len(earnings_m) > 0 else 0):.2f}%")
            st.markdown("""**Note on Scaled MAD:** The Median Absolute Deviation (MAD) is a robust measure of spread. It is scaled here by a factor of ~1.4826 to make it comparable to the standard deviation for data that is approximately normally distributed. The interval 'Median ± Scaled MAD' provides a robust alternative to 'Mean ± Standard Deviation'.""")


        # --- ECDF Plot ---
        st.subheader("Empirical Cumulative Distribution Function (ECDF)")
        ecdf_y_values = np.array([])
        ecdf_x_values = np.array([])
        median_val_ecdf = 0
        if not earnings.empty:
            ecdf_x_values = earnings.values 
            ecdf_y_values = (np.arange(1, len(earnings) + 1) / len(earnings)) 
            median_val_ecdf = earnings.median()

        if use_rank_filter: rank_info_for_ecdf = f"SGL Rank Range {sgl_rank_range_applied[0]}–{sgl_rank_range_applied[1]}"
        else: rank_info_for_ecdf = "All SGL Ranks (default)"
        ecdf_title_text = f"ECDF of {plot_title_status} Prize Money for {years_str_for_plot_title}{signed_filter_title_addon} ({rank_info_for_ecdf})"
        
        if len(ecdf_x_values) > 0 and len(ecdf_y_values) > 0:
            fig_ecdf = px.line(x=ecdf_x_values, y=ecdf_y_values, labels={'x': f'{earnings_column} ($)', 'y': 'Cumulative Proportion'}, title=ecdf_title_text)
            fig_ecdf.add_vline(x=median_val_ecdf, line_dash="dash", line_color="red", annotation_text=f"Median: ${median_val_ecdf:,.0f}")

            ecdf_plot_guarantee_info = get_relevant_guarantee_info_for_display(
                sgl_rank_range_applied[0], sgl_rank_range_applied[1], use_rank_filter, filtered_display, guarantee_map_dynamic # Pass filtered_display
            )
            if ecdf_plot_guarantee_info:
                guarantee_value_raw_ecdf = ecdf_plot_guarantee_info["value"] 
                proportion_below_guarantee = 0.0
                if len(ecdf_x_values) > 0:
                    if guarantee_value_raw_ecdf < ecdf_x_values[0]: proportion_below_guarantee = 0.0
                    elif guarantee_value_raw_ecdf >= ecdf_x_values[-1]: proportion_below_guarantee = 1.0
                    else:
                        idx = np.searchsorted(ecdf_x_values, guarantee_value_raw_ecdf, side='right')
                        if idx > 0: proportion_below_guarantee = ecdf_y_values[idx-1] 
                annotation_text_ecdf = f"{ecdf_plot_guarantee_info['label_short']}: ${guarantee_value_raw_ecdf:,.0f}<br>({proportion_below_guarantee*100:.1f}% at/below)"
                
                current_plot_ecdf_x_min = ecdf_x_values.min()
                current_plot_ecdf_x_max = ecdf_x_values.max()
                plot_range_delta = current_plot_ecdf_x_max - current_plot_ecdf_x_min if current_plot_ecdf_x_max > current_plot_ecdf_x_min else 1
                plot_display_ecdf_max_x = current_plot_ecdf_x_max + 0.05 * plot_range_delta
                plot_display_ecdf_min_x = current_plot_ecdf_x_min - 0.05 * plot_range_delta
                
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
            active_filter_guarantee_info = plot_display_guarantee_info 

            if active_filter_guarantee_info:
                target_guarantee_val_for_calc = active_filter_guarantee_info["value"] 
                
                # Base conditions for this section operate on 'filtered_display'
                snumtrn_cond_exposure = pd.Series(True, index=filtered_display.index)
                if 'Events Played' in filtered_display.columns: snumtrn_cond_exposure = (filtered_display['Events Played'] > 14)
                
                carprz_cond_exposure = pd.Series(True, index=filtered_display.index)
                if 'carprz' in filtered_display.columns: carprz_cond_exposure = (filtered_display['carprz'] < 15_000_000)

                earnings_cond_exposure = (filtered_display[earnings_column] < target_guarantee_val_for_calc)
                
                # The signed player filter is already applied to filtered_display if active
                # So, no need to re-apply it here explicitly for mask creation if filtered_display is used
                
                if not filtered_display.empty:
                    # Rank filter is already applied to filtered_display if use_rank_filter is True
                    # So players_in_range_df will inherently respect the rank filter and Signed Policy filter
                    final_mask_range_exposure = snumtrn_cond_exposure & carprz_cond_exposure & earnings_cond_exposure
                    players_in_range_df = filtered_display[final_mask_range_exposure] 
                else: 
                    players_in_range_df = pd.DataFrame(columns=filtered_display.columns) 
                
                num_players_in_range_exposed = len(players_in_range_df)
                
                total_exposure_val_in_range = 0
                if num_players_in_range_exposed > 0:
                    total_exposure_val_in_range = (target_guarantee_val_for_calc - players_in_range_df[earnings_column]).sum()

                exp_num_html = f"""<div style="background-color: #ffe0b3; color: #804000; border: 1px solid #ffcc80; border-radius: 0.25rem; padding: 0.5rem 1rem; font-weight: bold;">For Ranks {sgl_rank_range_applied[0]}-{sgl_rank_range_applied[1]} (vs G'tee ${target_guarantee_val_for_calc:,.0f}):<br># Players Exposed: {num_players_in_range_exposed}</div>"""
                exp_total_html = f"""<div style="background-color: #ffe0b3; color: #804000; border: 1px solid #ffcc80; border-radius: 0.25rem; padding: 0.5rem 1rem; font-weight: bold;">For Ranks {sgl_rank_range_applied[0]}-{sgl_rank_range_applied[1]} (vs G'tee ${target_guarantee_val_for_calc:,.0f}):<br>Total Exposure: ${total_exposure_val_in_range:,.0f}</div>"""
                st.markdown(f"""<div style='display: flex; flex-wrap: wrap; gap: 0.75rem; margin-top: 0.5rem; margin-bottom: 1rem;'>{exp_num_html}{exp_total_html}</div>""", unsafe_allow_html=True)
            elif use_rank_filter: 
                st.markdown(f"""<div style='margin-top: 0.5rem; margin-bottom: 1rem; font-style: italic;'>No specific ATP guarantee band strongly aligns with the selected rank range {sgl_rank_range_applied[0]}-{sgl_rank_range_applied[1]} for detailed exposure stats display here.</div>""", unsafe_allow_html=True)


    # --- Exposure Comparison Section ---
    st.markdown("---")
    exposure_type_label = "Adjusted" if use_adjusted_earnings else "Actual"
    st.header(f"{exposure_type_label} Exposure Comparison")
    st.markdown(f"---{'Only signed players included in this section if filter is active.' if use_signed_players_filter else ''}")

    # Determine which earnings column to use
    earnings_col_actual_section = 'Net Prize Money (2025 Adjusted)' if use_adjusted_earnings else 'Net Prize Money (Actual)'

    actual_exposure_years_to_calc = []
    if not use_adjusted_earnings:
        actual_exposure_years_to_calc = selected_years
    else:
        if selected_years:
            available_actual_rank_years_in_df = df['Year'].dropna().unique()
            actual_exposure_years_to_calc = [y for y in selected_years if y in available_actual_rank_years_in_df]

    snumtrn_exists_globally = 'Events Played' in df.columns
    carprz_exists_globally = 'carprz' in df.columns

    if not actual_exposure_years_to_calc:
        st.info("No Baseline Years selected or available for Exposure calculation based on current filters.")
    else:
        st.subheader(f"Exposure Analysis (for Year(s): {', '.join(map(str, sorted(set(actual_exposure_years_to_calc))))} using '{earnings_col_actual_section}')")

        if not snumtrn_exists_globally:
            st.warning("Column 'Events Played' not found. Cannot apply events filter.")
        if not carprz_exists_globally:
            st.warning("Column 'carprz' not found. Cannot apply career prize filter.")
        if use_signed_players_filter and not signed_policy_col_exists:
            st.warning("Column 'Signed Policy' not found. Cannot apply signed player filter.")

        results_exposure = []
        for year_val in sorted(set(actual_exposure_years_to_calc)):
            df_year = df[df['Year'] == year_val].copy()
            count0, exposure0 = 0, 0
            count1, exposure1 = 0, 0
            count2, exposure2 = 0, 0
            comment = None

            if df_year.empty:
                comment = f"No data for year {year_val}"
            else:
                base_mask0 = df_year['sglrank'].between(1, 100) & (df_year[earnings_col_actual_section] < guarantee_1_100)
                base_mask1 = df_year['sglrank'].between(101, 175) & (df_year[earnings_col_actual_section] < guarantee_101_175)
                base_mask2 = df_year['sglrank'].between(176, 250) & (df_year[earnings_col_actual_section] < guarantee_176_250)

                snumtrn_filter = pd.Series(True, index=df_year.index)
                if snumtrn_exists_globally and 'Events Played' in df_year.columns:
                    snumtrn_filter = df_year['Events Played'] > 14

                carprz_filter = pd.Series(True, index=df_year.index)
                if carprz_exists_globally and 'carprz' in df_year.columns:
                    carprz_filter = df_year['carprz'] < 15_000_000

                signed_filter = pd.Series(True, index=df_year.index)
                if use_signed_players_filter and signed_policy_col_exists:
                    signed_filter = df_year['Signed Policy'].astype(str).str.contains('P', na=False)

                mask0 = base_mask0 & snumtrn_filter & carprz_filter & signed_filter
                mask1 = base_mask1 & snumtrn_filter & carprz_filter & signed_filter
                mask2 = base_mask2 & snumtrn_filter & carprz_filter & signed_filter

                count0 = mask0.sum()
                count1 = mask1.sum()
                count2 = mask2.sum()

                if count0 > 0:
                    exposure0 = (guarantee_1_100 - df_year.loc[mask0, earnings_col_actual_section]).sum()
                if count1 > 0:
                    exposure1 = (guarantee_101_175 - df_year.loc[mask1, earnings_col_actual_section]).sum()
                if count2 > 0:
                    exposure2 = (guarantee_176_250 - df_year.loc[mask2, earnings_col_actual_section]).sum()

            results_exposure.append({
                "Year": year_val,
                "Count_1_100": count0, "Exposure_1_100": exposure0,
                "Count_101_175": count1, "Exposure_101_175": exposure1,
                "Count_176_250": count2, "Exposure_176_250": exposure2,
                "comment": comment
            })

        if results_exposure:
            exposure_df = pd.DataFrame(results_exposure).sort_values(by='Year')
            valid_plot_df = exposure_df[exposure_df['comment'].isna()].copy()

            if not valid_plot_df.empty:
                st.subheader(f"Trend of Exposure Players by Year ({exposure_type_label})")
                fig_counts = px.line(
                    valid_plot_df, x='Year',
                    y=['Count_1_100', 'Count_101_175', 'Count_176_250'],
                    labels={'value': 'Number of Players', 'Year': 'Year'},
                    markers=True,
                    title=f"Number of Players Below Threshold by Year ({exposure_type_label})"
                )
                fig_counts.update_layout(legend_title_text='Player Groups')
                st.plotly_chart(fig_counts)

                st.subheader(f"Trend of Exposure Amount by Year ({exposure_type_label})")
                fig_amounts = px.line(
                    valid_plot_df, x='Year',
                    y=['Exposure_1_100', 'Exposure_101_175', 'Exposure_176_250'],
                    labels={'value': 'Total Exposure ($)', 'Year': 'Year'},
                    markers=True,
                    title=f"Total Exposure Amount by Year ({exposure_type_label})"
                )
                fig_amounts.update_layout(legend_title_text='Exposure Groups', yaxis_tickformat='$,.0f')
                st.plotly_chart(fig_amounts)

            st.subheader("Detailed Exposure Summary:")
            for _, row in exposure_df.iterrows():
                st.markdown(f"**Year: {int(row['Year'])}**")
                if row["comment"]:
                    st.write(row["comment"])
                    st.markdown("---")
                    continue

                col0, col1, col2 = st.columns(3)
                col0.metric(f"Ranks 1–100", int(row["Count_1_100"]), f"${row['Exposure_1_100']:,.0f}")
                col1.metric(f"Ranks 101–175", int(row["Count_101_175"]), f"${row['Exposure_101_175']:,.0f}")
                col2.metric(f"Ranks 176–250", int(row["Count_176_250"]), f"${row['Exposure_176_250']:,.0f}")

                total_players = int(row["Count_1_100"] + row["Count_101_175"] + row["Count_176_250"])
                total_exposure = row["Exposure_1_100"] + row["Exposure_101_175"] + row["Exposure_176_250"]

                st.markdown(
                    f"""
                    <div style="background-color: #0f0f0f; border: 1px solid #ccc; border-radius: 6px;
                                padding: 10px; margin-top: 10px;">
                        <strong>Total:</strong><br>
                        Players Below Threshold: <strong>{total_players}</strong><br>
                        Total Exposure: <strong>${total_exposure:,.0f}</strong>
                    </div>
                    """, unsafe_allow_html=True
                )
                st.markdown("---")
        else:
            st.info("No data processed for exposure analysis.")

else:
    st.warning("Please upload the earnings data CSV file to proceed.")