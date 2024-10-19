import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the historical data
data = pd.read_csv('82lottery_data.csv')

# Convert color names to numerical labels
data['Color_Label'] = data['Color'].astype('category').cat.codes

# Extract features and labels
X = data[['Draw_ID']]
y = data['Color_Label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit UI
st.title('82Lottery Color Prediction Tool')

st.write(f'Model Accuracy: {accuracy:.2f}')

# Predict the next draw color
next_draw_id = data['Draw_ID'].max() + 1
predicted_label = model.predict([[next_draw_id]])[0]
predicted_color = data['Color'].astype('category').cat.categories[predicted_label]

st.write(f'Predicted Color for Draw {next_draw_id}: **{predicted_color}**')

# Display color frequency chart
st.write("Color Frequency in 82Lottery:")
st.bar_chart(data['Color'].value_counts())
