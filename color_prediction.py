import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load the Historical Data
data = pd.read_csv('82lottery_data.csv')

# Step 2: Data Preprocessing and Feature Engineering
# Convert color names to numerical labels
data['Color_Label'] = data['Color'].astype('category').cat.codes

# Extract features: Draw_ID as the only feature (you can expand features if needed)
X = data[['Draw_ID']]
y = data['Color_Label']

# Step 3: Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Display the Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Step 6: Predict the Next Draw Color
next_draw_id = data['Draw_ID'].max() + 1
predicted_label = model.predict([[next_draw_id]])[0]

# Convert numerical label back to color name
predicted_color = data['Color'].astype('category').cat.categories[predicted_label]
print(f'Predicted Color for Draw {next_draw_id}: {predicted_color}')

# Step 7: Visualize Color Frequencies
plt.figure(figsize=(8, 5))
data['Color'].value_counts().plot(kind='bar', color=['red', 'green', 'blue'])
plt.title('Color Frequency in 82Lottery')
plt.xlabel('Color')
plt.ylabel('Frequency')
plt.show()
