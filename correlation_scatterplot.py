import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import chardet
df = pd.read_csv("human_sentiment_assessmentv2.csv", encoding="latin-1")

#removes whitespaces from sentiment catergory
df['Human_Sentiment_Catergory'] = df['Human_Sentiment_Catergory'].str.strip()
df['Sentiment_Category'] = df['Sentiment_Category'].str.strip()

# Define function to map sentiment scores to categories based on ranges
def categorize_sentiment(score):
    if score <= -0.5:
        return 'Negative'
    elif -0.49 <= score <= 0.49:
        return 'Neutral'
    else:
        return 'Positive'

# Apply function to create numerical sentiment categories
df['Human_Sentiment_Catergory'] = df['Human_Sentiment_Score'].apply(categorize_sentiment)
df['Sentiment_Category'] = df['Sentiment_Score'].apply(categorize_sentiment)


# Define **high-contrast color palette** mapped to sentiment categories
color_palette = {
    'Negative': 'blue', 'Neutral': 'black', 'Positive': 'red',
    'Human_Negative': 'purple', 'Human_Neutral': 'gray', 'Human_Positive': 'green',
    'VADER_Negative': 'orange', 'VADER_Neutral': 'yellow', 'VADER_Positive': 'cyan'
}

### **🔥 PLOT 1: Human vs. VADER Sentiment Scores**
plt.figure(figsize=(7, 5))
scatter = sns.scatterplot(
    x=df['Human_Sentiment_Score'], 
    y=df['Sentiment_Score'], 
    hue=df['Human_Sentiment_Catergory'],  # Use category names, not direct colors
    palette={'Negative': 'blue', 'Neutral': 'black', 'Positive': 'red'},  
    alpha=0.6,  
    edgecolor='black'
)
plt.xlabel('Human Sentiment Score')
plt.ylabel('VADER Sentiment Score')
plt.title('Scatter Plot: Human vs. VADER Sentiment Scores')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.axvline(0, color='black', linestyle='--', linewidth=1)

# Move legend to top-right white space
plt.legend(title="Human Sentiment Score", loc='upper right', bbox_to_anchor=(1.3, 1))
plt.show()

### **🔥 PLOT 2: Human Sentiment Category vs. VADER Score**
plt.figure(figsize=(7, 5))
scatter = sns.scatterplot(
    x=df['Sentiment_Score'], 
    y=df['Human_Sentiment_Catergory'],  
    hue=df['Human_Sentiment_Catergory'],  # Use category names
    palette={'Negative': 'purple', 'Neutral': 'gray', 'Positive': 'green'},  
    alpha=0.6,  
    edgecolor='black'
)
plt.xlabel('VADER Sentiment Score')
plt.ylabel('Human Sentiment Category')
plt.title('Scatter Plot: VADER Sentiment Score vs. Human Sentiment Category')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.axvline(0, color='black', linestyle='--', linewidth=1)

# Move legend to top-right white space
plt.legend(title="Human Sentiment Category", loc='upper right', bbox_to_anchor=(1.3, 1))
plt.show()

### **🔥 PLOT 3: VADER Sentiment Score vs. VADER Sentiment Category**
plt.figure(figsize=(7, 5))
scatter = sns.scatterplot(
    x=df['Sentiment_Score'], 
    y=df['Sentiment_Category'],  
    hue=df['Sentiment_Category'],  # Use category names
    palette={'Negative': 'orange', 'Neutral': 'yellow', 'Positive': 'cyan'},  
    alpha=0.6,  
    edgecolor='black'
)
plt.xlabel('VADER Sentiment Score')
plt.ylabel('VADER Sentiment Category')
plt.title('Scatter Plot: VADER Sentiment Score vs. VADER Sentiment Category')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.axvline(0, color='black', linestyle='--', linewidth=1)

# Move legend to top-right white space
plt.legend(title="VADER Sentiment Category", loc='upper right', bbox_to_anchor=(1.3, 1))
plt.show()