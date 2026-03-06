import streamlit as st
import pandas as pd
import joblib

model = joblib.load("best_blinkit_model.pkl")

st.title("Blinkit Sales Prediction")

item_weight = st.number_input("Item Weight",0.0,50.0,10.0)

item_fat_content = st.selectbox(
    "Item Fat Content",
    ["Low Fat","Regular"]
)

item_visibility = st.slider("Item Visibility",0.0,0.5,0.05)

item_type = st.selectbox(
    "Item Type",
    ["Dairy","Soft Drinks","Meat","Fruits and Vegetables",
     "Household","Baking Goods","Snack Foods","Frozen Foods",
     "Breakfast","Health and Hygiene","Hard Drinks",
     "Canned","Breads","Starchy Foods","Others","Seafood"]
)

rating = st.slider("Rating",1.0,5.0,3.5)

outlet_size = st.selectbox(
    "Outlet Size",
    ["Small","Medium","High"]
)

outlet_location_type = st.selectbox(
    "Outlet Location Type",
    ["Tier 1","Tier 2","Tier 3"]
)

outlet_type = st.selectbox(
    "Outlet Type",
    ["Supermarket Type1","Supermarket Type2","Supermarket Type3","Grocery Store"]
)

outlet_age = st.number_input("Outlet Age",1,50,10)

if st.button("Predict"):

    input_data = pd.DataFrame({
        "Item Weight":[item_weight],
        "Item Fat Content":[item_fat_content],
        "Item Visibility":[item_visibility],
        "Item Type":[item_type],
        "Rating":[rating],
        "Outlet Size":[outlet_size],
        "Outlet Location Type":[outlet_location_type],
        "Outlet Type":[outlet_type],
        "Outlet Age":[outlet_age]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted Sales: {prediction[0]:.2f}")
