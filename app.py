import streamlit as st
import requests

# =========================
# CONFIGURATION
# =========================
API_URL = "http://127.0.0.1:8000/predict"
TIMEOUT = 10

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Wave Energy Predictor", page_icon="🌊", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #0077b6;'>🌊 Wave Energy Converter</h1>
    <h4 style='text-align: center; color: gray;'>Enter WEC Buoy Coordinates to Predict Total Power Output</h4>
    <hr>
""", unsafe_allow_html=True)

# =========================
# COORDINATE INPUT FORM
# =========================
st.subheader("📍 Enter X and Y Coordinates for Each WEC Buoy")

st.markdown("""
| | **X Coordinate (m)** | **Y Coordinate (m)** |
|---|---|---|
""")

input_data = {}

# Header row
header_col0, header_col1, header_col2 = st.columns([1, 3, 3])
header_col0.markdown("**WEC #**")
header_col1.markdown("**X Coordinate (m)**")
header_col2.markdown("**Y Coordinate (m)**")

st.divider()

# Input rows for all 49 WECs
for i in range(1, 50):
    col0, col1, col2 = st.columns([1, 3, 3])

    with col0:
        st.markdown(f"<br><b>WEC {i}</b>", unsafe_allow_html=True)

    with col1:
        input_data[f"X{i}"] = st.number_input(
            label=f"X{i}",
            min_value=0.0,
            max_value=2000.0,
            value=0.0,
            step=0.01,
            format="%.2f",
            key=f"X{i}",
            label_visibility="collapsed"
        )

    with col2:
        input_data[f"Y{i}"] = st.number_input(
            label=f"Y{i}",
            min_value=0.0,
            max_value=2000.0,
            value=0.0,
            step=0.01,
            format="%.2f",
            key=f"Y{i}",
            label_visibility="collapsed"
        )

st.divider()

# =========================
# PREDICT BUTTON + RESULT
# =========================
col_left, col_center, col_right = st.columns([2, 2, 2])

with col_center:
    predict_btn = st.button("⚡ Predict Total Power", type="primary", use_container_width=True)

st.divider()

if predict_btn:
    # Validate - check if all inputs are zero (likely not filled)
    all_zero = all(v == 0.0 for v in input_data.values())
    if all_zero:
        st.warning("⚠️ All coordinates are 0. Please enter valid WEC positions before predicting.")
    else:
        with st.spinner("🔄 Sending data to prediction API..."):
            try:
                response = requests.post(API_URL, json=input_data, timeout=TIMEOUT)
                result = response.json()

                if "prediction" in result:
                    power = result["prediction"]

                    # Result display
                    st.success("✅ Prediction Complete!")

                    r1, r2, r3 = st.columns(3)
                    with r1:
                        st.metric("🔋 Total Power (W)", f"{power:,.2f}")
                    with r2:
                        st.metric("⚡ Total Power (kW)", f"{power/1000:,.3f}")
                    with r3:
                        st.metric("🌊 Total Power (MW)", f"{power/1e6:,.4f}")

                elif "error" in result:
                    st.error(f"❌ API Error: {result['error']}")
                else:
                    st.warning("⚠️ Unexpected response.")
                    st.json(result)

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure FastAPI is running:\n\n`uvicorn main:app --reload`")
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. API might be slow or unreachable.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")

# =========================
# RAW INPUT VIEWER
# =========================
with st.expander("🔍 View Raw Input JSON Being Sent to API"):
    st.json(input_data)