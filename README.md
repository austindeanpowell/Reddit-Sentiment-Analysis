# Reddit-Sentiment-Analysis

This repository contains a Python-based sentiment analysis tool designed to extract and evaluate sentiment from Reddit comments using NLP (Natural Language Processing), leveraging the VADER sentiment analyzer (from the NLTK library) and the PRAW (Python Reddit API Wrapper).

**The Reddit Sentiment Analyzer project provides an easy-to-use pipeline for:**

Extracting Reddit comments using the Reddit API (via PRAW)
Analyzing comment sentiment using VADER
Categorizing sentiment as Positive, Neutral, or Negative
Visualizing sentiment results with data analytics techniques (Pandas, Seaborn, Matplotlib)

**SET UP VIRTUAL ENVIROMENT**
python3 -m venv venv
source venv/bin/activate

**INSTALL DEPENDENCIES**
pip install -r requirements.txt


** Tools & Technologies Used:**
PRAW	Extracting comments from Reddit API
NLTK	Sentiment Analysis using VADER
Pandas	Data manipulation and analysis
Matplotlib & Seaborn	Visualization of results
Python-dotenv	Securely handling API credentials


The script will output a CSV file (reddit_sentiment_analysis_DATE.csv) containing:

Comment IDs
Original text
Sentiment scores
Sentiment categories (Positive, Neutral, Negative)

This project generates intuitive visualizations including:

Sentiment distribution plots
Correlation heatmaps (Human vs. AI sentiment scores)
Confusion matrices for detailed analysis of sentiment predictions
