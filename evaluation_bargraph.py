import matplotlib.pyplot as plt

# Define evaluation metrics and their scores
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
scores = [0.427386, 0.525810, 0.513659, 0.433689]  # Example values from your evaluation

# Create a bar chart
plt.figure(figsize=(7, 5))  # Sets the figure size
plt.bar(metrics, scores, color=['blue', 'green', 'orange', 'red'])  # Creates the bars

# Set y-axis limits (since metrics range from 0 to 1)
plt.ylim(0, 1)

# Label the axes
plt.ylabel("Score")
plt.title("Evaluation Metrics: Human vs. VADER Sentiment Classification")

# Add text labels above bars for clarity
for i, score in enumerate(scores):
    plt.text(i, score + 0.02, round(score, 2), ha='center', fontsize=12)

# Show the chart
plt.show()
