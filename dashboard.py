import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
from sklearn.ensemble import HistGradientBoostingClassifier

# ==================================================
# PAGE CONFIGURATION (UI IMPROVEMENT)
# ==================================================
st.set_page_config(
    page_title="Drainage Bottleneck Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
# 🌊 Urban Drainage Bottleneck & Flood-Path Analyzer  
### *AI-assisted identification of flood-prone choke points in urban drainage networks*
""")

st.info(
    "📌 This system uses **Gradient Boosting AI** to identify critical drainage bottlenecks "
    "by analyzing terrain, flow capacity, rainfall, and blockage patterns."
)

# ==================================================
# DATA LOADING + AI MODEL (UNCHANGED LOGIC)
# ==================================================
@st.cache_data
def load_data():
    df = pd.read_csv('india_drainage_10k_dataset.csv')

    features = [
        'elevation_m', 'slope_percent', 'pipe_diameter_m',
        'capacity_cumecs', 'rainfall_mm_hr',
        'flow_velocity_mps', 'blockage_percent'
    ]

    model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    model.fit(df[features], df['is_bottleneck'])

    def suggest(row):
        if row['blockage_percent'] > 70:
            return "Urgent: Desilting Required"
        if row['slope_percent'] < 0.5:
            return "Slope Correction Needed"
        if row['pipe_diameter_m'] < 1.0:
            return "Infrastructure Upgrade: Upsize Pipe"
        return "Install Sensor Monitoring"

    df['Recommendation'] = df.apply(suggest, axis=1)
    return df, features

df, ml_features = load_data()

# ==================================================
# SIDEBAR (BETTER GUIDANCE)
# ==================================================
st.sidebar.header("🔍 Analysis Controls")
st.sidebar.markdown(
    "Use the filters below to **drill down** into high-risk areas. "
    "All analytics update dynamically."
)

city_list = ["All"] + list(df['city'].unique())
selected_city = st.sidebar.selectbox("🏙️ Select City", city_list)

risk_filter = st.sidebar.slider(
    "⚠️ Minimum Flood Risk Score",
    0, 100, 50,
    help="Higher values show only more severe flood-risk segments"
)

filtered_df = df[df['flood_risk_score'] >= risk_filter]
if selected_city != "All":
    filtered_df = filtered_df[filtered_df['city'] == selected_city]

# ==================================================
# KPI METRICS (BETTER LABELING)
# ==================================================
st.markdown("## 📊 Key Risk Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Drainage Segments Analyzed",
    len(filtered_df),
    help="Number of drainage segments considered under current filters"
)

col2.metric(
    "Detected Bottlenecks",
    int(filtered_df['is_bottleneck'].sum()),
    help="Segments classified as bottlenecks by the AI model"
)

col3.metric(
    "Average Blockage (%)",
    f"{filtered_df['blockage_percent'].mean():.1f}%",
    help="Higher values indicate restricted water flow"
)

col4.metric(
    "Severe Risk Segments",
    len(filtered_df[filtered_df['flood_risk_score'] > 80]),
    help="Segments with extreme flood risk"
)

# ==================================================
# AI INTERPRETABILITY SECTION
# ==================================================
st.markdown("---")
st.markdown("## 🧠 AI Decision Insights")

st.success(
    "The charts below explain **WHY** the AI flags certain drainage segments as bottlenecks."
)

ai_col1, ai_col2 = st.columns(2)

with ai_col1:
    corr = (
        filtered_df[ml_features + ['is_bottleneck']]
        .corr()['is_bottleneck']
        .sort_values(ascending=False)
        .drop('is_bottleneck')
    )

    fig_ai = px.bar(
        corr,
        x=corr.values,
        y=corr.index,
        orientation='h',
        title="Key Factors Influencing Bottleneck Formation",
        labels={'x': 'Influence Strength', 'y': 'Drainage Parameter'},
        color=corr.values,
        color_continuous_scale='Reds'
    )

    st.plotly_chart(fig_ai, use_container_width=True)

    st.caption(
        "🔎 **Interpretation**: Parameters with higher influence values contribute more "
        "strongly to bottleneck formation."
    )

with ai_col2:
    fig_bubble = px.scatter(
        filtered_df,
        x="rainfall_mm_hr",
        y="blockage_percent",
        size="flood_risk_score",
        color="is_bottleneck",
        hover_name="segment_id",
        title="Rainfall vs Blockage – Flood Risk Interaction",
        color_continuous_scale='RdYlGn_r'
    )

    st.plotly_chart(fig_bubble, use_container_width=True)

    st.caption(
        "💡 **Insight**: High rainfall combined with high blockage sharply increases flood risk."
    )

# ==================================================
# GEOSPATIAL VISUALIZATION
# ==================================================
st.markdown("---")
st.markdown(f"## 🗺️ Spatial Bottleneck Visualization — {selected_city}")

st.warning(
    "🔴 Red points indicate AI-detected bottlenecks. "
    "🟢 Green points indicate normal drainage segments."
)

filtered_df['color'] = filtered_df['is_bottleneck'].apply(
    lambda x: [255, 0, 0, 150] if x == 1 else [0, 200, 0, 120]
)

view_state = pdk.ViewState(
    latitude=filtered_df['latitude'].mean(),
    longitude=filtered_df['longitude'].mean(),
    zoom=11,
    pitch=45
)

layer = pdk.Layer(
    "ScatterplotLayer",
    filtered_df,
    get_position='[longitude, latitude]',
    get_color='color',
    get_radius=100,
    pickable=True
)

st.pydeck_chart(
    pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "Segment: {segment_id}\nRisk Score: {flood_risk_score}"}
    )
)

# ==================================================
# SUPPORTING ANALYTICS
# ==================================================
st.markdown("---")
st.markdown("## 📈 Supporting Risk Analytics")

c1, c2 = st.columns(2)

with c1:
    fig_land = px.box(
        filtered_df,
        x="land_use",
        y="flood_risk_score",
        color="land_use",
        title="Flood Risk Distribution by Land Use"
    )
    st.plotly_chart(fig_land, use_container_width=True)

with c2:
    fig_scatter = px.scatter(
        filtered_df,
        x="blockage_percent",
        y="capacity_cumecs",
        color="is_bottleneck",
        title="Blockage vs Drainage Capacity"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ==================================================
# ENHANCED MITIGATION RECOMMENDATIONS (UI + INSIGHTS)
# ==================================================
st.markdown("---")
st.markdown("## 🛠️ Priority Mitigation Recommendations (Action-Oriented View)")

st.success(
    "This section converts AI predictions into **clear operational actions** "
    "for city engineers and decision-makers."
)

# -------------------------------
# Prepare mitigation dataframe
# -------------------------------
mitigation_df = (
    filtered_df[filtered_df['is_bottleneck'] == 1]
    .sort_values('flood_risk_score', ascending=False)
    .copy()
)

# Severity classification
def severity(score):
    if score >= 85:
        return "🔴 Critical"
    elif score >= 70:
        return "🟠 High"
    else:
        return "🟡 Moderate"

mitigation_df["Severity"] = mitigation_df["flood_risk_score"].apply(severity)

# Action icons
action_icon_map = {
    "Urgent: Desilting Required": "🧹 Desilting",
    "Infrastructure Upgrade: Upsize Pipe": "🔧 Pipe Upgrade",
    "Install Sensor Monitoring": "📡 Monitoring",
    "Slope Correction Needed": "📐 Slope Fix"
}

mitigation_df["Action Type"] = mitigation_df["Recommendation"].map(action_icon_map)

# Explanation column (human-friendly)
def explain(row):
    if row["Recommendation"] == "Urgent: Desilting Required":
        return "Heavy blockage is restricting water flow. Immediate cleaning required."
    if row["Recommendation"] == "Infrastructure Upgrade: Upsize Pipe":
        return "Pipe capacity is insufficient for current rainfall conditions."
    if row["Recommendation"] == "Install Sensor Monitoring":
        return "Area shows moderate risk; monitoring recommended for early warning."
    return "Drainage slope is too low, slowing down water discharge."

mitigation_df["Why This Action?"] = mitigation_df.apply(explain, axis=1)

# -------------------------------
# QUICK INSIGHT CARDS
# -------------------------------
c1, c2, c3 = st.columns(3)

c1.metric("🔴 Critical Segments", (mitigation_df["Severity"] == "🔴 Critical").sum())
c2.metric("🧹 Desilting Needed", (mitigation_df["Recommendation"] == "Urgent: Desilting Required").sum())
c3.metric("🔧 Structural Fixes", (mitigation_df["Recommendation"] == "Infrastructure Upgrade: Upsize Pipe").sum())

st.markdown("---")

# -------------------------------
# Interactive Filters
# -------------------------------
rec_filter = st.multiselect(
    "🎯 Filter by Recommended Action",
    mitigation_df["Recommendation"].unique(),
    default=mitigation_df["Recommendation"].unique()
)

severity_filter = st.multiselect(
    "⚠️ Filter by Severity Level",
    mitigation_df["Severity"].unique(),
    default=mitigation_df["Severity"].unique()
)

filtered_mitigation = mitigation_df[
    (mitigation_df["Recommendation"].isin(rec_filter)) &
    (mitigation_df["Severity"].isin(severity_filter))
]

# -------------------------------
# Display Table (Enhanced)
# -------------------------------
st.dataframe(
    filtered_mitigation[
        [
            "segment_id",
            "city",
            "Severity",
            "flood_risk_score",
            "blockage_percent",
            "Action Type",
            "Why This Action?"
        ]
    ].head(25),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "🧠 **How to read this table:**\n"
    "- Severity shows urgency level\n"
    "- Action Type is the operational fix\n"
    "- Explanation translates AI output into engineering insight"
)
