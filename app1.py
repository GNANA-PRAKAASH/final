import streamlit as st
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))

st.set_page_config(page_title="Zomato Rating Prediction", page_icon=":smiley:")

def predict(features):
    '''
    Function to predict the restaurant rating.
    '''
    final_features = np.array(features).reshape(1, -1)
    prediction = model.predict(final_features)
    return round(prediction[0], 1)

def main():
    st.title("Zomato Rating Prediction")
    st.markdown("### Please enter the following details:")

    online_order = st.selectbox("Online Order (1 for Yes, 0 for No)", [1, 0])
    book_table = st.selectbox("Book Table (1 for Yes, 0 for No)", [1, 0])
    votes = st.text_input("Votes", placeholder="Enter number of votes")
    restaurant_type = st.text_input("Restaurant Type", placeholder="Enter restaurant type")
    dishes_liked = st.text_input("Dishes Liked", placeholder="Enter number of dishes liked")
    cuisines = st.text_input("Cuisines", placeholder="Enter number of cuisines")
    cost = st.text_input("Cost For 2 Person", placeholder="Enter cost for 2 persons")
    
    if st.button("  Predict Rating"):
        features = [online_order, book_table, votes,  restaurant_type, dishes_liked, cuisines, cost]
        prediction = predict(features)
        st.success(f"Predicted Rating: {prediction}")

if __name__ == "__main__":
    main()
