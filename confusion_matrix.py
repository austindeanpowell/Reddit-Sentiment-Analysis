import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load dataset
df = pd.read_csv("human_sentiment_assessmentv2.csv", encoding="latin-1")

# Strip whitespace from sentiment categories (fixes potential mismatches)
df['Human_Sentiment_Catergory'] = df['Human_Sentiment_Catergory'].str.strip()
df['Sentiment_Category'] = df['Sentiment_Category'].str.strip()

# Define function to map sentiment scores to 3 categories (-1, 0, 1)
def categorize_sentiment(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

# Apply function to create categorical sentiment labels
df['Human_Sentiment_Catergory'] = df['Human_Sentiment_Score'].apply(categorize_sentiment)
df['Sentiment_Category'] = df['Sentiment_Score'].apply(categorize_sentiment)

# Define **numeric mapping** for 3 categories (-1, 0, 1)
category_mapping = {
    'Negative': -1, 
    'Neutral': 0, 
    'Positive': 1
}

# Apply numeric mapping to create numerical equivalents for sentiment categories
df['Human_Sentiment_Category_Num'] = df['Human_Sentiment_Catergory'].map(category_mapping).astype(int)
df['Sentiment_Category_Num'] = df['Sentiment_Category'].map(category_mapping).astype(int)

# Ensure mapping worked
print(df[['Human_Sentiment_Catergory', 'Human_Sentiment_Category_Num']].head(5))
print(df[['Sentiment_Category', 'Sentiment_Category_Num']].head(5))

# Calculate correlation matrix (keeping raw scores)
score_correlation = df[['Human_Sentiment_Score', 'Sentiment_Score']].corr()
category_correlation = df[['Human_Sentiment_Category_Num', 'Sentiment_Category_Num']].corr()

# Print correlation results for reference
print("\nCorrelation between Human & VADER Sentiment Scores:")
print(score_correlation)
print("\nCorrelation between Human & VADER Sentiment Categories (Numeric):")
print(category_correlation)

### **🔥 PLOT 1: Heatmap of Sentiment Score Correlation**
plt.figure(figsize=(6, 5))
sns.heatmap(score_correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1, linecolor='black')
plt.title("Correlation Heatmap: Human vs. VADER Sentiment Scores")
plt.xlabel("Sentiment Score Type")
plt.ylabel("Sentiment Score Type")
plt.show()

### **🔥 PLOT 2: Confusion Matrix of Sentiment Categories**
# Create Confusion Matrix
conf_matrix = confusion_matrix(df['Human_Sentiment_Category_Num'], df['Sentiment_Category_Num'])

# Convert to DataFrame for visualization
conf_df = pd.DataFrame(
    conf_matrix, 
    index=['Human Negative', 'Human Neutral', 'Human Positive'], 
    columns=['VADER Negative', 'VADER Neutral', 'VADER Positive']
)

# Display Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(conf_df, annot=True, cmap='Blues', fmt='d', linewidths=1, linecolor='black')
plt.title("Confusion Matrix: Human vs. VADER Sentiment Categories")
plt.xlabel("VADER Sentiment Category")
plt.ylabel("Human Sentiment Category")
plt.show()
