import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Toronto vs Vancouver Crime Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load data ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_parquet("../data/merged_crime_data.parquet")
    clusters = pd.read_csv("../output/models/neighbourhood_clusters.csv")
    return df, clusters

@st.cache_resource
def load_models():
    rf       = joblib.load("../output/models/random_forest.pkl")
    le_city  = joblib.load("../output/models/le_city.pkl")
    le_month = joblib.load("../output/models/le_month.pkl")
    le_neigh = joblib.load("../output/models/le_neighbourhood.pkl")
    return rf, le_city, le_month, le_neigh

df, clusters = load_data()
rf, le_city, le_month, le_neigh = load_models()

COLORS = {'Toronto': '#2196F3', 'Vancouver': '#FF5722'}

# ── Sidebar ───────────────────────────────────────────────────────
st.sidebar.title("🔍 Crime Analysis")
st.sidebar.markdown("Toronto vs Vancouver (2016–2025)")
page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "📈 Trends",
    "🗺️ Crime Map",
    "🤖 ML Predictor"
])

st.sidebar.markdown("---")
city_filter = st.sidebar.multiselect(
    "Filter by City",
    options=['Toronto', 'Vancouver'],
    default=['Toronto', 'Vancouver']
)
year_range = st.sidebar.slider(
    "Year Range",
    min_value=2016, max_value=2025,
    value=(2016, 2025)
)

# Apply filters
df_filtered = df[
    df['city'].isin(city_filter) &
    df['year'].between(year_range[0], year_range[1])
]

# ════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("🏙️ Toronto vs Vancouver Crime Analysis")
    st.markdown("Comparative analysis of crime patterns across two major Canadian cities (2016–2025)")

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Incidents", f"{len(df_filtered):,}")
    with col2:
        tor_count = len(df_filtered[df_filtered['city'] == 'Toronto'])
        st.metric("Toronto", f"{tor_count:,}")
    with col3:
        van_count = len(df_filtered[df_filtered['city'] == 'Vancouver'])
        st.metric("Vancouver", f"{van_count:,}")
    with col4:
        years = year_range[1] - year_range[0] + 1
        st.metric("Years Analyzed", years)

    st.markdown("---")

    # Crime group distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Crime Group Distribution")
        group_pct = (
            df_filtered.groupby(['city', 'crime_group'])
            .size().reset_index(name='count')
        )
        group_pct['pct'] = group_pct.groupby('city')['count'].transform(
            lambda x: x / x.sum() * 100
        )
        pivot = group_pct.pivot(index='city', columns='crime_group', values='pct').fillna(0)
        col_order = [c for c in ['Violent Crime','Property Crime','Property Damage','Other'] if c in pivot.columns]
        pivot = pivot[col_order]

        fig, ax = plt.subplots(figsize=(8, 4))
        pivot.plot(kind='bar', stacked=True, ax=ax,
                   color=['#E53935','#1E88E5','#FDD835','#43A047'],
                   edgecolor='white')
        for container in ax.containers:
            labels = [f'{v:.1f}%' if v > 2 else '' for v in container.datavalues]
            ax.bar_label(container, labels=labels, label_type='center',
                         fontsize=9, color='white', fontweight='bold')
        ax.set_xlabel("")
        ax.set_ylabel("Percentage (%)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(title="Crime Group", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Crime Type Breakdown")
        crime_counts = (
            df_filtered.groupby(['city', 'crime_type'])
            .size().reset_index(name='count')
        )
        crime_counts['pct'] = crime_counts.groupby('city')['count'].transform(
            lambda x: x / x.sum() * 100
        )
        all_types = sorted(crime_counts['crime_type'].unique())
        tor = crime_counts[crime_counts['city'] == 'Toronto'].set_index('crime_type')['pct']
        van = crime_counts[crime_counts['city'] == 'Vancouver'].set_index('crime_type')['pct']
        tor_vals = [tor.get(t, 0) for t in all_types]
        van_vals = [van.get(t, 0) for t in all_types]

        fig, ax = plt.subplots(figsize=(8, 4))
        x = range(len(all_types))
        w = 0.35
        ax.bar([i - w/2 for i in x], tor_vals, w, label='Toronto', color=COLORS['Toronto'])
        ax.bar([i + w/2 for i in x], van_vals, w, label='Vancouver', color=COLORS['Vancouver'])
        ax.set_xticks(list(x))
        ax.set_xticklabels(all_types, rotation=25, ha='right', fontsize=8)
        ax.set_ylabel("% of Total Crime")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Neighbourhood risk table
    st.markdown("---")
    st.subheader("🏘️ Neighbourhood Risk Levels (K-Means Clustering)")
    col1, col2 = st.columns(2)
    for col, city in zip([col1, col2], ['Toronto', 'Vancouver']):
        with col:
            st.markdown(f"**{city}**")
            city_clusters = clusters[clusters['city'] == city][
                ['neighbourhood', 'risk_level', 'total_crimes']
            ].sort_values('total_crimes', ascending=False).head(15)
            city_clusters.columns = ['Neighbourhood', 'Risk Level', 'Total Crimes']
            city_clusters['Total Crimes'] = city_clusters['Total Crimes'].apply(lambda x: f"{x:,}")

            def color_risk(val):
                colors = {'High Risk': 'background-color: #ffcccc',
                          'Medium Risk': 'background-color: #ffe0b2',
                          'Low Risk': 'background-color: #c8e6c9'}
                return colors.get(val, '')

            st.dataframe(
                city_clusters.style.map(color_risk, subset=['Risk Level']),
                use_container_width=True, 
                hide_index=True
            )

# ════════════════════════════════════════════════════════════════
# PAGE 2 — Trends
# ════════════════════════════════════════════════════════════════
elif page == "📈 Trends":
    st.title("📈 Crime Trends Analysis")

    tab1, tab2, tab3 = st.tabs(["Yearly Trend", "Hourly Pattern", "Monthly Seasonality"])

    with tab1:
        yearly = df_filtered.groupby(['city', 'year']).size().reset_index(name='count')
        fig, ax = plt.subplots(figsize=(11, 5))
        for city, grp in yearly.groupby('city'):
            ax.plot(grp['year'], grp['count'], marker='o',
                    color=COLORS[city], label=city, linewidth=2.5)
        ax.set_title("Total Crime Incidents per Year", fontsize=14, fontweight='bold')
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Incidents")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        hourly = df_filtered.groupby(['city', 'hour']).size().reset_index(name='count')
        hourly['pct'] = hourly.groupby('city')['count'].transform(lambda x: x / x.sum() * 100)
        pivot_hour = hourly.pivot(index='city', columns='hour', values='pct')
        fig, ax = plt.subplots(figsize=(14, 3))
        sns.heatmap(pivot_hour, ax=ax, cmap='YlOrRd', annot=True, fmt='.1f',
                    annot_kws={'size': 8}, linewidths=0.3)
        ax.set_title("Crime Distribution by Hour of Day (%)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Hour of Day (0–23)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        month_order = ['January','February','March','April','May','June',
                       'July','August','September','October','November','December']
        monthly = df_filtered.groupby(['city', 'month']).size().reset_index(name='count')
        monthly['month'] = pd.Categorical(monthly['month'], categories=month_order, ordered=True)
        monthly = monthly.sort_values('month')
        fig, ax = plt.subplots(figsize=(13, 5))
        for city, grp in monthly.groupby('city'):
            ax.plot(grp['month'].astype(str), grp['count'],
                    marker='o', color=COLORS[city], label=city, linewidth=2.5)
        ax.set_title("Monthly Crime Seasonality", fontsize=14, fontweight='bold')
        ax.set_ylabel("Number of Incidents")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        ax.tick_params(axis='x', rotation=30)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════════════
# PAGE 3 — Crime Map
# ════════════════════════════════════════════════════════════════
elif page == "🗺️ Crime Map":
    st.title("🗺️ Interactive Crime Map")

    map_city = st.selectbox("Select City", ['Toronto', 'Vancouver'])
    map_type = st.radio("Map Type", ["Heatmap", "Crime Type Layers"], horizontal=True)

    city_coords = {'Toronto': [43.7, -79.4], 'Vancouver': [49.27, -123.11]}
    city_zoom   = {'Toronto': 11, 'Vancouver': 12}

    city_df = df_filtered[df_filtered['city'] == map_city].dropna(subset=['latitude','longitude'])

    if map_type == "Heatmap":
        m = folium.Map(location=city_coords[map_city],
                       zoom_start=city_zoom[map_city],
                       tiles='CartoDB positron')
        sample = city_df.sample(min(10000, len(city_df)), random_state=42)
        HeatMap(sample[['latitude','longitude']].values.tolist(),
                radius=8, blur=10).add_to(m)
        st_folium(m, width=1000, height=500)

    else:
        crime_colors = {
            'Violent Crime': 'red', 'Property Crime': 'blue',
            'Property Damage': 'orange', 'Other': 'gray'
        }
        m = folium.Map(location=city_coords[map_city],
                       zoom_start=city_zoom[map_city],
                       tiles='CartoDB positron')
        for group, color in crime_colors.items():
            group_df = city_df[city_df['crime_group'] == group]
            if len(group_df) == 0:
                continue
            fg = folium.FeatureGroup(name=f"{group} ({len(group_df):,})", show=True)
            sample = group_df.sample(min(1500, len(group_df)), random_state=42)
            for _, row in sample.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=3, color=color, fill=True, fill_opacity=0.5,
                    popup=folium.Popup(
                        f"<b>{row['crime_type']}</b><br>"
                        f"{row['neighbourhood']}<br>"
                        f"Year: {row['year']} | Hour: {row['hour']}h",
                        max_width=200
                    )
                ).add_to(fg)
            fg.add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        st_folium(m, width=1000, height=500)

# ════════════════════════════════════════════════════════════════
# PAGE 4 — ML Predictor
# ════════════════════════════════════════════════════════════════
elif page == "🤖 ML Predictor":
    st.title("🤖 Crime Type Predictor")
    st.markdown("Use the Random Forest model to predict whether a crime is **Property** or **Violent** based on location and time.")

    col1, col2 = st.columns(2)

    with col1:
        pred_city = st.selectbox("City", ['Toronto', 'Vancouver'])
        pred_hour = st.slider("Hour of Day", 0, 23, 12)
        pred_month = st.selectbox("Month", [
            'January','February','March','April','May','June',
            'July','August','September','October','November','December'
        ])

    with col2:
        city_neighbourhoods = sorted(
            df[df['city'] == pred_city]['neighbourhood'].unique()
        )
        pred_neigh = st.selectbox("Neighbourhood", city_neighbourhoods)

    if st.button("🔍 Predict", type="primary"):
        try:
            city_enc  = le_city.transform([pred_city])[0]
            month_enc = le_month.transform([pred_month])[0]
            neigh_enc = le_neigh.transform([pred_neigh])[0]

            X_pred = pd.DataFrame([[city_enc, pred_hour, month_enc, neigh_enc]],
                                   columns=['city_enc','hour','month_enc','neigh_enc'])
            pred = rf.predict(X_pred)[0]
            prob = rf.predict_proba(X_pred)[0]
            conf = max(prob) * 100

            color = "🔴" if pred == "Violent Crime" else "🔵"
            st.success(f"{color} **Predicted: {pred}** (Confidence: {conf:.1f}%)")

            # Probability bar
            fig, ax = plt.subplots(figsize=(6, 2))
            classes = rf.classes_
            colors  = ['#E53935' if c == 'Violent Crime' else '#1E88E5' for c in classes]
            ax.barh(classes, prob * 100, color=colors, edgecolor='white')
            ax.set_xlabel("Probability (%)")
            ax.set_xlim(0, 100)
            for i, v in enumerate(prob * 100):
                ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        except ValueError as e:
            st.error(f"Prediction error: {e}")

    st.markdown("---")
    st.subheader("📊 Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "75%")
    col2.metric("Property Crime F1", "0.76")
    col3.metric("Violent Crime F1", "0.75")

    st.image("../output/figures/model_results.png",
             caption="Confusion Matrix & Feature Importance", width="stretch")