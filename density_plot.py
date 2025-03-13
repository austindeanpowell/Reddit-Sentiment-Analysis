import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
category_mapping = {'Negative': -1, 'Neutral': 0, 'Positive': 1}

# Apply numeric mapping to create numerical equivalents for sentiment categories
df['Human_Sentiment_Category_Num'] = df['Human_Sentiment_Catergory'].map(category_mapping).astype(int)
df['Sentiment_Category_Num'] = df['Sentiment_Category'].map(category_mapping).astype(int)

# Ensure mapping worked
print(df[['Human_Sentiment_Score', 'Sentiment_Score']].head(5))

# Create a figure with **2 subplots** for Score Distribution & Category Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

### **🔥 PLOT 1: Density Plot (KDE) of Sentiment Scores**
sns.kdeplot(df['Human_Sentiment_Score'], label="Human Sentiment Score", shade=True, color="blue", alpha=0.6, ax=axes[0])
sns.kdeplot(df['Sentiment_Score'], label="VADER Sentiment Score", shade=True, color="red", alpha=0.6, ax=axes[0])

axes[0].set_xlabel("Sentiment Score")
axes[0].set_ylabel("Density")
axes[0].set_title("Density Plot: Human vs. VADER Sentiment Score Distribution")
axes[0].legend(title="Sentiment Type")
axes[0].grid(True)

### **🔥 PLOT 2: Sentiment Category Distribution (Histogram)**
sns.histplot(df['Human_Sentiment_Catergory'], label="Human Sentiment Category", color="blue", alpha=0.6, stat="density", ax=axes[1])
sns.histplot(df['Sentiment_Category'], label="VADER Sentiment Category", color="red", alpha=0.6, stat="density", ax=axes[1])

axes[1].set_xlabel("Sentiment Category")
axes[1].set_ylabel("Density")
axes[1].set_title("Sentiment Category Distribution: Human vs. VADER")
axes[1].legend(title="Sentiment Type")
axes[1].grid(True)

# Adjust layout for better spacing
plt.tight_layout()

# Show both plots
plt.show()
