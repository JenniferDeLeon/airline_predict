import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

airline_df = pd.read_csv("airline.csv")

# Set up the title and description of the app
st.title('Airline Satisfaction Prediction') 
st.write("Gain insights into passenger experiences and improve satisfaction through data analysis and surveys.")

# Display an image of airlines
st.image('airline.jpg', width = 400)

with st.expander("What can you do with this app?"):
    st.markdown("📝 **Fill Out a Survey:** Provide a form for users to fill out their airline satisfaction details.")
    st.markdown("🌟 **Make Data-Driven Decisions:** Use insights to guide improvements in customer experience.")
    st.markdown("🛠️ **Interactive Features:** Explore data with fully interactive charts and summaries!")

# Load models
with open('decision_tree_airline.pickle', 'rb') as f:
    dt_model = pickle.load(f)

features = [
    'customer_type', 'age', 'type_of_travel', 'class',
    'flight_distance', 'seat_comfort', 'departure_arrival_time_convenient',
    'food_and_drink', 'gate_location', 'inflight_wifi_service',
    'inflight_entertainment', 'online_support', 'ease_of_online_booking',
    'on-board_service', 'leg_room_service', 'baggage_handling',
    'checkin_service', 'cleanliness',
    'departure_delay_in_minutes', 'arrival_delay_in_minutes']

# Sidebar for user input
st.sidebar.header("Airline Customer Satisfaction Survey")
st.sidebar.subheader("Part 1: Customer Details")
st.sidebar.write("Provide information on the customer flying.")

customer_type = st.sidebar.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])
type_of_travel = st.sidebar.selectbox("Is the customer travelling for business or personal reasons", ["Business travel", "Personal Travel"])
class_type = st.sidebar.selectbox("Class", ["Eco", "Eco Plus", "Business"])
age = st.sidebar.number_input("How old is the customer?", min_value = 0, value = 25)

st.sidebar.subheader("Part 2: Flight Details")
st.sidebar.write("Provide details about the customer's flight details.")

distance = st.sidebar.number_input("How far is the customer flying in miles?", min_value = 0, value = 50)
departure_delay = st.sidebar.number_input("How many minutes was the customer's departure delayed? (Enter 0 if not delayed)", min_value = 0, value = 0)
arrival_delay = st.sidebar.number_input("How many minutes was the customer's arrival delayed? (Enter 0 if not delayed)", min_value = 0, value = 0)

st.sidebar.subheader("Part 3: Customer Experience")
st.sidebar.write("Provide details about the customer's flight experience and satisfaction.")

seat_comfort = st.sidebar.radio("How comfortable was the seat for the customer? (1-5 stars)", [1, 2, 3, 4, 5], horizontal=True)
convenient_time = st.sidebar.radio("Was the departure/arrival time convenient for the customer? (1-5 stars)", [1, 2, 3, 4, 5], horizontal=True)
food = st.sidebar.radio("How would the customer rate the food and drink? (1-5 stars)", [1, 2, 3, 4, 5], horizontal=True)
gate_location = st.sidebar.radio("How would the customer rate the gate location? (1-5 stars)", [1, 2, 3, 4, 5], horizontal=True)
wifi = st.sidebar.radio("How would the customer rate the inflight wifi service? (1-5 stars)", [1, 2, 3, 4, 5], horizontal=True)
entertainment = st.sidebar.radio("How would the customer rate the inflight entertainment? (1-5 stars)", [1, 2, 3, 4, 5], horizontal=True)
online_support = st.sidebar.radio("How would the customer rate online support? (1-5 stars)", [1, 2, 3, 4, 5], horizontal=True)
online_booking = st.sidebar.radio("How easy was online booking for the customer? (1-5 stars)", [1, 2, 3, 4, 5], horizontal=True)
onboard_service = st.sidebar.radio("How would the customer rate the onboard service? (1-5 stars)", [1, 2, 3, 4, 5], horizontal=True)
leg_room = st.sidebar.radio("How would the customer rate the leg room service? (1-5 stars)", [1, 2, 3, 4, 5], horizontal=True)
baggage_handling = st.sidebar.radio("How would the customer rate baggage handling? (1-5 stars)", [1, 2, 3, 4, 5], horizontal=True)
checkin_service = st.sidebar.radio("How would the customer rate the check-in service? (1-5 stars)", [1, 2, 3, 4, 5], horizontal=True)
cleanliness = st.sidebar.radio("How would the customer rate cleanliness? (1-5 stars)", [1, 2, 3, 4, 5], horizontal=True)
online_boarding = st.sidebar.radio("How would the customer rate online boarding? (1-5 stars)", [1, 2, 3, 4, 5], horizontal=True)

# ----------------- Build Input DataFrame -----------------
input_dict = {
    "customer_type": customer_type,
    "age": age,
    "type_of_travel": type_of_travel,
    "class": class_type,
    "flight_distance": distance,
    "seat_comfort": seat_comfort,
    "departure_arrival_time_convenient": convenient_time,
    "food_and_drink": food,
    "gate_location": gate_location,
    "inflight_wifi_service": wifi,
    "inflight_entertainment": entertainment,
    "online_support": online_support,
    "ease_of_online_booking": online_booking,
    "on-board_service": onboard_service,
    "leg_room_service": leg_room,
    "baggage_handling": baggage_handling,
    "checkin_service": checkin_service,
    "cleanliness": cleanliness,
    "departure_delay_in_minutes": departure_delay,
    "arrival_delay_in_minutes": arrival_delay
}

st.header("Prediction of Customer Satisfaction (Decision Tree)")

# I used ChatGPT here to help me figure out the function for the session state banner and the if statements
if "form_submitted" not in st.session_state:
    st.session_state["form_submitted"] = False
if "input_features" not in st.session_state:
    st.session_state["input_features"] = {}

# Reset form_submitted if inputs change 
# Used ChatGPT here to help me figure out how to reset the session_state if the user changed any input
if st.session_state.get("input_features") != input_dict:
    st.session_state["form_submitted"] = False

# Sidebar button
if st.sidebar.button("Predict"):
    st.session_state["form_submitted"] = True
    st.session_state["input_features"] = input_dict

# Show info if button not clicked yet
if not st.session_state["form_submitted"]:
    st.info("ℹ️ Please fill out the survey form in the sidebar and click **Predict** to see the satisfaction prediction.")
else:
    # Retrieve input features from session state
    input_dict = st.session_state["input_features"]
    input_df = pd.DataFrame([input_dict])

    # One-hot encode categorical columns
    categorical_cols = ['customer_type', 'type_of_travel', 'class']
    input_df = pd.get_dummies(input_df, columns=categorical_cols)

    # Align columns with model
    input_df = input_df.reindex(columns=dt_model.feature_names_in_, fill_value=0)

    # Make prediction
    prediction = dt_model.predict(input_df)[0]
    proba = dt_model.predict_proba(input_df)[0]
    predicted_class_index = dt_model.classes_.tolist().index(prediction)
    confidence_perc = proba[predicted_class_index] * 100

    # Display result
    st.subheader("**Prediction Result**")
    st.subheader(f"Your predicted satisfaction level is **{prediction}**")
    st.write(f"With a confidence of **{confidence_perc:.1f}%**")

    total_customers = len(airline_df)

# I used AI here to clean up the syntax and formatting 

    with st.expander("Customer Type Comparison"):
        selected_count = (airline_df['customer_type'] == customer_type).sum()
        customer_type_percentage = (selected_count / total_customers) * 100
        customer_type_percentage = round(customer_type_percentage, 1)
    
        st.write("**Customer Type:** Your selection: " + customer_type)
        st.write("Percentage of our fliers with this selection: **{}%**".format(customer_type_percentage))

    with st.expander("Type of Travel Comparison"):
        type_of_travel_count = (airline_df['type_of_travel'] == type_of_travel).sum()
        type_of_travel_percentage = (type_of_travel_count / total_customers) * 100
        type_of_travel_percentage = round(type_of_travel_percentage, 1)
    
        st.write("**Type of Travel:** Your selection: " + type_of_travel)
        st.write("Percentage of our fliers with this selection: **{}%**".format(type_of_travel_percentage))

    with st.expander("Flight Class Comparison"):
        class_count = (airline_df['class'] == class_type).sum()
        class_percentage = (class_count / total_customers) * 100
        class_percentage = round(class_percentage, 1)
    
        st.write("**Flight Class:** Your selection: " + class_type)
        st.write("Percentage of our fliers with this selection: **{}%**".format(class_percentage))
    
    with st.expander("Age Group Comparison"):
        age_bins = [0, 18, 30, 45, 60, 100]
        age_labels = ['0-17', '18-30', '31-45', '46-60', '60+']

        # Create age_group column
        airline_df['age_group'] = pd.cut(airline_df['age'], bins=age_bins, labels=age_labels, right=False)
    
        # Determine user's age group
        user_age_group = None
        for i in range(len(age_bins)-1):
            if age >= age_bins[i] and age < age_bins[i+1]:
                user_age_group = age_labels[i]
                break
    
        # Calculate percentage
        age_group_count = (airline_df['age_group'] == user_age_group).sum()
        age_group_percentage = round((age_group_count / total_customers) * 100, 1)
    
        st.write("**Age Group:** Your selection: " + str(age))
        st.write("Your selected age group: " + str(user_age_group))  # <-- show only user's group
        st.write("Percentage of our fliers in this age group: **{}%**".format(age_group_percentage))