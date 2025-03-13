from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


# Manually extracting values from the confusion matrix image
conf_matrix_values = np.array([[25, 8, 21],
                               [15, 47, 84],
                               [9, 1, 31]])

# Defining the actual (human-labeled) and predicted (VADER-labeled) values
labels = ['Negative', 'Neutral', 'Positive']

# Creating a dataset for evaluation
y_true = []
y_pred = []

for i, actual_label in enumerate(labels):  # Human sentiment categories
    for j, predicted_label in enumerate(labels):  # VADER sentiment categories
        y_true.extend([actual_label] * conf_matrix_values[i, j])
        y_pred.extend([predicted_label] * conf_matrix_values[i, j])

# Compute accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

# Generate classification report
report = classification_report(y_true, y_pred, target_names=labels)

# Store metrics in a DataFrame for display
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Score": [accuracy, precision, recall, f1]
})

# Display results
print(metrics_df)

report


# Define metrics and scores
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
scores = [0.427386, 0.525810, 0.513659, 0.433689]

# Create bar chart
plt.figure(figsize=(7, 5))
plt.bar(metrics, scores, color=['blue', 'green', 'orange', 'red'])
plt.ylim(0, 1)  # Ensuring scores are within 0-1 range
plt.ylabel("Score")
plt.title("Evaluation Metrics: Human vs. VADER Sentiment Classification")

# Add text labels above bars
for i, score in enumerate(scores):
    plt.text(i, score + 0.02, round(score, 2), ha='center', fontsize=12)

# Show chart
plt.show()
