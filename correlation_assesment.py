import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import chardet
df = pd.read_csv("human_sentiment_assessmentv2.csv", encoding="latin-1")

#define function 
catergory_mapping = {'Negative': -0.5, 'Negative':- 1, 'Neutral':0, 'Positive':0.5, 'Positive':1  }

df['Human_Sentiment_Catergory_Num'] = df['Human_Sentiment_Catergory'].map(catergory_mapping)
df['Sentiment_Category_Num'] = df['Sentiment_Category'].map(catergory_mapping)

# Calculate correlation matrix (keeping VADER_Score raw)
score_correlation = df[['Human_Sentiment_Score', 'Sentiment_Score']].corr()
category_correlation = df[['Human_Sentiment_Catergory_Num', 'Sentiment_Category_Num']].corr()

# Print correlation results
print("Human Catergory & Score Assessment vs VADER's Catergory & Score Assessment:")
print(score_correlation)

print("\nCorrelation between Human & VADER Sentiment Categories (Converted to Numeric):")
print(category_correlation)

# Generate heatmap for visual representation
plt.figure(figsize=(7, 5))
sns.heatmap(score_correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1, linecolor='black')
plt.title('Correlation: Human vs VADER Sentiment Scores')
plt.show()

plt.figure(figsize=(7, 5))
sns.heatmap(category_correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1, linecolor='black')
plt.title('Correlation: Human vs VADER Sentiment Categories (Converted to Numeric)')
plt.show()


# Define color mapping for sentiment categories
category_colors = {'Negative': 'red', 'Neutral': 'gray', 'Positive': 'green'}

plt.figure(figsize=(7, 5))
sns.scatterplot(
    x=df['Human_Sentiment_Score'], 
    y=df['Sentiment_Score'], 
    hue=df['Human_Sentiment_Catergory'],  # Use human category for color
    palette=category_colors,  
    alpha=0.6,  # Adjust transparency
    edgecolor='black'  # Make points more visible
)

# Add labels & title
plt.xlabel('Human Sentiment Score')
plt.ylabel('VADER Sentiment Score')
plt.title('Scatter Plot: Human vs. VADER Sentiment Scores: ')

# Add reference lines at 0
plt.axhline(0, color='black', linestyle='--', linewidth=1)  # Neutral line (horizontal)
plt.axvline(0, color='black', linestyle='--', linewidth=1)  # Neutral line (vertical)

# Show plot
plt.legend(title="Sentiment Category")  # Add legend
plt.show()
# Grouping Data for Stakeholders - Count Matches
#category_comparison = pd.crosstab(df['Human_Sentiment_Catergory'], df['Sentiment_Category'])
#print("\nHuman vs VADER Sentiment Category Comparison:")
#print(category_comparison)

