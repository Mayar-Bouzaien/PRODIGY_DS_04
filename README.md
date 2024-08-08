# PRODIGY_DS_04
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import kaggle

kaggle.api.dataset_download_file('jp797498e/twitter-entity-sentiment-analysis', 
                                 file_name='twitter_training.csv', 
                                 path='C:/Users/LENOVO/Downloads/') 

import zipfile
with zipfile.ZipFile('C:/Users/LENOVO/Downloads/twitter_training.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('C:/Users/LENOVO/Downloads/') 
df = pd.read_csv('twitter_training.csv', header=None)
df.columns = ['ID', 'Entity', 'Sentiment', 'Text']
print(df.head())
df.drop_duplicates(subset=['ID'], inplace=True)

# Sentiment Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Sentiment', data=df)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Sentiment Distribution by Entity
plt.figure(figsize=(14, 8))
sns.countplot(x='Entity', hue='Sentiment', data=df)
plt.title('Sentiment Distribution by Entity')
plt.xlabel('Entity')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

# Top 10 Entities by Sentiment
top_entities = df['Entity'].value_counts().nlargest(10).index
df_top_entities = df[df['Entity'].isin(top_entities)]

plt.figure(figsize=(14, 8))
sns.countplot(x='Entity', hue='Sentiment', data=df_top_entities)
plt.title('Top 10 Entities by Sentiment')
plt.xlabel('Entity')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Word Cloud for Positive Sentiment
positive_text = ' '.join(df[df['Sentiment'] == 'Positive']['Text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Positive Sentiment')
plt.axis('off')
plt.show()

# Word Cloud for Negative Sentiment
negative_text = ' '.join(df[df['Sentiment'] == 'Negative']['Text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Negative Sentiment')
plt.axis('off')
plt.show()




