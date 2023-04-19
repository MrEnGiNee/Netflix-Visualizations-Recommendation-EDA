#!/usr/bin/env python
# coding: utf-8

# ## Netflix Visualizations, Recommendation, EDA

# Netflix is an application that keeps growing bigger and faster with its popularity, shows and content. This is an EDA or a story telling through its data along with a content-based recommendation system and a wide range of different graphs and visuals.

# In[4]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt


# #### Loading the dataset

# In[5]:


netflix_overall=pd.read_csv('C:/Users/dell/Downloads/Netflix/netflix_titles.csv')
netflix_overall.head()


# Therefore, it is clear that the dataset contains 12 columns for exploratory analysis.

# In[6]:


netflix_overall.count()


# In[7]:


netflix_shows=netflix_overall[netflix_overall['type']=='TV Show']


# In[8]:


netflix_movies=netflix_overall[netflix_overall['type']=='Movie']


# #### Analysis of Movies vs TV Shows

# In[9]:


sns.set(style="darkgrid")
ax = sns.countplot(x="type", data=netflix_overall, palette="Set2")


# If a producer wants to release some content, which month must he do so?( Month when least amount of content is added)

# In[10]:


netflix_date = netflix_shows[['date_added']].dropna()
netflix_date['year'] = netflix_date['date_added'].apply(lambda x : x.split(', ')[-1])
netflix_date['month'] = netflix_date['date_added'].apply(lambda x : x.lstrip().split(' ')[0])

month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][::-1]
df = netflix_date.groupby('year')['month'].value_counts().unstack().fillna(0)[month_order].T
plt.figure(figsize=(10, 7), dpi=200)
plt.pcolor(df, cmap='afmhot_r', edgecolors='white', linewidths=2) # heatmap
plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns, fontsize=7, fontfamily='serif')
plt.yticks(np.arange(0.5, len(df.index), 1), df.index, fontsize=7, fontfamily='serif')

plt.title('Netflix Contents Update', fontsize=12, fontfamily='calibri', fontweight='bold', position=(0.20, 1.0+0.02))
cbar = plt.colorbar()

cbar.ax.tick_params(labelsize=8) 
cbar.ax.minorticks_on()
plt.show()


# If the latest year 2019 is considered, January and December were the months when comparatively much less content was released.Therefore, these months may be a good choice for the success of a new release!

# #### Movie ratings analysis

# In[11]:


plt.figure(figsize=(12,10))
sns.set(style="darkgrid")
ax = sns.countplot(x="rating", data=netflix_movies, palette="Set2", order=netflix_movies['rating'].value_counts().index[0:15])


# The largest count of movies are made with the 'TV-MA' rating."TV-MA" is a rating assigned by the TV Parental Guidelines to a television program that was designed for mature audiences only.
# 
# Second largest is the 'TV-14' stands for content that may be inappropriate for children younger than 14 years of age.
# 
# Third largest is the very popular 'R' rating.An R-rated film is a film that has been assessed as having material which may be unsuitable for children under the age of 17 by the Motion Picture Association of America; the MPAA writes "Under 17 requires accompanying parent or adult guardian".

# #### Analysing IMDB ratings to get top rated movies on Netflix

# In[13]:


imdb_ratings=pd.read_csv('C:/Users/dell/Downloads/Netflix/IMDb ratings.csv',usecols=['weighted_average_vote'])
imdb_titles=pd.read_csv('C:/Users/dell/Downloads/Netflix/IMDb movies.csv', usecols=['title','year','genre'])
ratings = pd.DataFrame({'Title':imdb_titles.title,
                    'Release Year':imdb_titles.year,
                    'Rating': imdb_ratings.weighted_average_vote,
                    'Genre':imdb_titles.genre})
ratings.drop_duplicates(subset=['Title','Release Year','Rating'], inplace=True)
ratings.shape


# Performing inner join on the ratings dataset and netflix dataset to get the content that has both ratings on IMDB and are available on Netflix.

# In[14]:


ratings.dropna()
joint_data=ratings.merge(netflix_overall,left_on='Title',right_on='title',how='inner')
joint_data=joint_data.sort_values(by='Rating', ascending=False)


# #### Top rated 10 movies on Netflix are:

# In[15]:


import plotly
import plotly.express as px
top_rated=joint_data[0:10]
fig =px.sunburst(
    top_rated,
    path=['title','country'],
    values='Rating',
    color='Rating')
fig.show()


# Countries with highest rated content.

# In[16]:


country_count=joint_data['country'].value_counts().sort_values(ascending=False)
country_count=pd.DataFrame(country_count)
topcountries=country_count[0:11]
topcountries


# In[17]:


import plotly.express as px
data = dict(
    number=[1063,619,135,60,44,41,40,40,38,35],
    country=["United States", "India", "United Kingdom", "Canada", "Spain",'Turkey','Philippines','France','South Korea','Australia'])
fig = px.funnel(data, x='number', y='country')
fig.show()


# #### Year wise analysis

# In[18]:


plt.figure(figsize=(12,10))
sns.set(style="darkgrid")
ax = sns.countplot(y="release_year", data=netflix_movies, palette="Set2", order=netflix_movies['release_year'].value_counts().index[0:15])


# So, 2017 and 2018 was the year when most of the movies were released.

# In[19]:


countries={}
netflix_movies['country']=netflix_movies['country'].fillna('Unknown')
cou=list(netflix_movies['country'])
for i in cou:
    #print(i)
    i=list(i.split(','))
    if len(i)==1:
        if i in list(countries.keys()):
            countries[i]+=1
        else:
            countries[i[0]]=1
    else:
        for j in i:
            if j in list(countries.keys()):
                countries[j]+=1
            else:
                countries[j]=1


# In[20]:


countries_fin={}
for country,no in countries.items():
    country=country.replace(' ','')
    if country in list(countries_fin.keys()):
        countries_fin[country]+=no
    else:
        countries_fin[country]=no
        
countries_fin={k: v for k, v in sorted(countries_fin.items(), key=lambda item: item[1], reverse= True)}


# #### TOP 10 MOVIE CONTENT CREATING COUNTRIES

# In[21]:


plt.figure(figsize=(8,8))
ax = sns.barplot(x=list(countries_fin.keys())[0:10],y=list(countries_fin.values())[0:10])
ax.set_xticklabels(list(countries_fin.keys())[0:10],rotation = 90)


# #### Analysis of duration of movies

# In[22]:


netflix_movies['duration']=netflix_movies['duration'].str.replace(' min','')
netflix_movies['duration']=netflix_movies['duration'].astype(str).astype(str)
netflix_movies['duration']


# In[23]:


from collections import Counter

genres=list(netflix_movies['listed_in'])
gen=[]

for i in genres:
    i=list(i.split(','))
    for j in i:
        gen.append(j.replace(' ',""))
g=Counter(gen)


# #### WordCloud for Genres

# In[32]:


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image

text = list(set(gen))
plt.rcParams['figure.figsize'] = (13, 13)

#assigning shape to the word cloud
mask = np.array(Image.open('C:/Users/dell/Downloads/Netflix/Star.png'))
wordcloud = WordCloud(max_words=1000000,background_color="white",mask=mask).generate(str(text))

plt.imshow(wordcloud,interpolation="bilinear")
plt.axis("off")
plt.show()


# #### Lollipop plot of Genres vs their count on Netflix

# In[33]:


g={k: v for k, v in sorted(g.items(), key=lambda item: item[1], reverse= True)}


fig, ax = plt.subplots()

fig = plt.figure(figsize = (10, 10))
x=list(g.keys())
y=list(g.values())
ax.vlines(x, ymin=0, ymax=y, color='green')
ax.plot(x,y, "o", color='maroon')
ax.set_xticklabels(x, rotation = 90)
ax.set_ylabel("Count of movies")
# set a title
ax.set_title("Genres");


# Therefore, it is clear that international movies, dramas and comedies are the top three genres that have the highest amount of content on Netflix.

# #### Analysis of TV SERIES on Netflix

# In[34]:


countries1={}
netflix_shows['country']=netflix_shows['country'].fillna('Unknown')
cou1=list(netflix_shows['country'])
for i in cou1:
    #print(i)
    i=list(i.split(','))
    if len(i)==1:
        if i in list(countries1.keys()):
            countries1[i]+=1
        else:
            countries1[i[0]]=1
    else:
        for j in i:
            if j in list(countries1.keys()):
                countries1[j]+=1
            else:
                countries1[j]=1


# In[35]:



countries_fin1={}
for country,no in countries1.items():
    country=country.replace(' ','')
    if country in list(countries_fin1.keys()):
        countries_fin1[country]+=no
    else:
        countries_fin1[country]=no
        
countries_fin1={k: v for k, v in sorted(countries_fin1.items(), key=lambda item: item[1], reverse= True)}


# #### Most content creating countries

# In[36]:


# Set the width and height of the figure
plt.figure(figsize=(15,15))

# Add title
plt.title("Content creating countries")

# Bar chart showing average arrival delay for Spirit Airlines flights by month
sns.barplot(y=list(countries_fin1.keys()), x=list(countries_fin1.values()))

# Add label for vertical axis
plt.ylabel("Arrival delay (in minutes)")


# Naturally, United States has the most content that is created on netflix in the tv series category.

# In[38]:


features=['title','duration']
durations= netflix_shows[features]

durations['no_of_seasons']=durations['duration'].str.replace(' Season','')

#durations['no_of_seasons']=durations['no_of_seasons'].astype(str).astype(int)
durations['no_of_seasons']=durations['no_of_seasons'].str.replace('s','')


# In[39]:


durations['no_of_seasons']=durations['no_of_seasons'].astype(str).astype(int)


# #### TV shows with largest number of seasons

# In[40]:


t=['title','no_of_seasons']
top=durations[t]

top=top.sort_values(by='no_of_seasons', ascending=False)


# In[41]:


top20=top[0:20]
top20.plot(kind='bar',x='title',y='no_of_seasons', color='red')


# Thus, NCIS, Grey's Anatomy and Supernatural are amongst the tv series that have highest number of seasons.

# #### Lowest number of seasons

# In[42]:


bottom=top.sort_values(by='no_of_seasons')
bottom=bottom[20:50]

import plotly.graph_objects as go

fig = go.Figure(data=[go.Table(header=dict(values=['Title', 'No of seasons']),
                 cells=dict(values=[bottom['title'],bottom['no_of_seasons']],fill_color='lavender'))
                     ])
fig.show()


# These are some binge-worthy shows that are short and have only one season.

# In[43]:


genres=list(netflix_shows['listed_in'])
gen=[]

for i in genres:
    i=list(i.split(','))
    for j in i:
        gen.append(j.replace(' ',""))
g=Counter(gen)


# #### Word Cloud for Genres

# A word cloud is an image made of words that together resemble a cloudy shape.

# In[44]:


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

text = list(set(gen))

mask=np.array(Image.open('C:/Users/dell/Downloads/Netflix/Netflix.png'))
wordcloud = WordCloud(max_words=1000000,background_color="black",mask=mask).generate(str(text))
plt.rcParams['figure.figsize'] = (13, 13)
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis("off")
plt.show()


# In[45]:


us_series_data=netflix_shows[netflix_shows['country']=='United States']


# In[46]:


oldest_us_series=us_series_data.sort_values(by='release_year')[0:20]


# In[47]:


fig = go.Figure(data=[go.Table(header=dict(values=['Title', 'Release Year'],fill_color='paleturquoise'),
                 cells=dict(values=[oldest_us_series['title'],oldest_us_series['release_year']],fill_color='pink'))
                     ])
fig.show()


# Above table shows the oldest US tv shows on Netflix.

# In[48]:


newest_us_series=us_series_data.sort_values(by='release_year', ascending=False)[0:50]


# In[49]:


fig = go.Figure(data=[go.Table(header=dict(values=['Title', 'Release Year'],fill_color='yellow'),
                 cells=dict(values=[newest_us_series['title'],newest_us_series['release_year']],fill_color='lavender'))
                     ])
fig.show()


# The above are latest released US television shows!

# #### Content in France

# In[50]:


netflix_fr=netflix_overall[netflix_overall['country']=='France']
nannef=netflix_fr.dropna()
import plotly.express as px
fig = px.treemap(nannef, path=['country','director'],
                  color='director', hover_data=['director','title'],color_continuous_scale='Purples')
fig.show()


# It is very interesting to note that the content in France is very rational. There is no director in the data who has a large number of movies. In my opinion, it shows how different directors are given a chance to showcase their talents. What do you think?

# In[51]:


newest_fr_series=netflix_fr.sort_values(by='release_year', ascending=False)[0:20]


# In[52]:


newest_fr_series


# In[53]:


fig = go.Figure(data=[go.Table(header=dict(values=['Title', 'Release Year']),
                 cells=dict(values=[newest_fr_series['title'],newest_fr_series['release_year']]))
                     ])
fig.show()


# #### Top Duration

# In[54]:


topdirs=pd.value_counts(netflix_overall['duration'])
fig = go.Figure([go.Bar(x=topdirs.index, y=topdirs.values , text=topdirs.values,marker_color='indianred')])
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.show()


# It can be inferred that having one season is the most preferred duration.

# ### Recommendation System (Content Based)

# The TF-IDF(Term Frequency-Inverse Document Frequency (TF-IDF) ) score is the frequency of a word occurring in a document, down-weighted by the number of documents in which it occurs. This is done to reduce the importance of words that occur frequently in plot overviews and therefore, their significance in computing the final similarity score.

# In[55]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[56]:


#removing stopwords
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
netflix_overall['description'] = netflix_overall['description'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(netflix_overall['description'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape


# Here, The Cosine similarity score is used since it is independent of magnitude and is relatively easy and fast to calculate.
# 
# 

# In[57]:


# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[58]:


indices = pd.Series(netflix_overall.index, index=netflix_overall['title']).drop_duplicates()


# In[59]:


def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return netflix_overall['title'].iloc[movie_indices]


# This recommendation is just based on the Plot.

# In[60]:


get_recommendations('Mortel')


# It is seen that the model performs well, but is not very accurate.Therefore, more metrics are added to the model to improve performance.

# ### Content based filtering on multiple metrics
Content based filtering on the following factors:

Title
Cast
Director
Listed in
Plot
# Filling null values with empty string.

# In[61]:


filledna=netflix_overall.fillna('')
filledna.head(2)


# Cleaning the data - making all the words lower case

# In[62]:


def clean_data(x):
        return str.lower(x.replace(" ", ""))


# Identifying features on which the model is to be filtered.

# In[63]:


features=['title','director','cast','listed_in','description']
filledna=filledna[features]


# In[64]:


for feature in features:
    filledna[feature] = filledna[feature].apply(clean_data)
    
filledna.head(5)


# Creating a "soup" or a "bag of words" for all rows.

# In[65]:


def create_soup(x):
    return x['title']+ ' ' + x['director'] + ' ' + x['cast'] + ' ' +x['listed_in']+' '+ x['description']


# In[66]:


filledna['soup'] = filledna.apply(create_soup, axis=1)


# From here on, the code is basically similar to the upper model except the fact that count vectorizer is used instead of tfidf.

# In[67]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(filledna['soup'])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


# In[68]:


filledna=filledna.reset_index()
indices = pd.Series(filledna.index, index=filledna['title'])


# In[69]:


def get_recommendations_new(title, cosine_sim=cosine_sim):
    title=title.replace(' ','').lower()
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return netflix_overall['title'].iloc[movie_indices]


# In[70]:


get_recommendations_new('PK', cosine_sim2)


# In[71]:


get_recommendations_new('Peaky Blinders', cosine_sim2)


# In[72]:


get_recommendations_new('The Hook Up Plan', cosine_sim2)


# In[73]:


get_recommendations_new('Raees', cosine_sim2)


# In[74]:


get_recommendations_new('Taken', cosine_sim2)

