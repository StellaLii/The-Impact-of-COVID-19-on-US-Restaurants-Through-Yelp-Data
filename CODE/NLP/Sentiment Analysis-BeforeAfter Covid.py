#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Run in terminal or command prompt
# python3 -m spacy download en
import numpy as np
import pandas as pd
import re, nltk, spacy, gensim
import pandas as pd
import zipfile

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint
# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Sentiment
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# # 1. Read Data

# In[ ]:


# file_name = "OneDrive_1_3-26-2021.zip"
# with zipfile.ZipFile(file_name, 'r') as zip:
#     # printing all the contents of the zip file
#     print(zip.printdir())


# In[27]:


#zf = zipfile.ZipFile('OneDrive_1_3-26-2021.zip') 
df1 = pd.read_csv('merge_before_covid_2021.csv')
df2 = pd.read_csv('merge_after_covid_2021.csv')


# In[28]:


df = pd.concat([df1,df2])


# In[3]:


# df1 = df1.sample(frac=0.03, replace=True, random_state=1)
# df2 = df2.sample(frac=0.05, replace=True, random_state=1)
df = pd.concat([df1,df2])


# In[31]:


print(df1.shape)
print(df2.shape)


# In[ ]:


572395+331839


# In[5]:


df.shape


# In[ ]:


#df = df.sample(frac=0.5, replace=True, random_state=1)


# In[6]:


df.columns


# # 2. Preprocessing

# In[32]:


# CAT_useful
bins = [0,1,2,3, 5,  np.inf]
df['CAT_useful'] = pd.cut(df['useful'], bins)
df['CAT_useful'].value_counts()
# CAT_funny
bins = [0,1,2,3, 5,  np.inf]
df['CAT_funny'] = pd.cut(df['funny'], bins)
df['CAT_funny'].value_counts()
# CAT_cool
bins = [0,1,2,3, 5,  np.inf]
df['CAT_cool'] = pd.cut(df['cool'], bins)
df['CAT_cool'].value_counts()
# CAT_review_count
bins = [0,100,250,500, 1000, np.inf]
df['CAT_review_count'] = pd.cut(df['review_count'], bins)
df['CAT_review_count'].value_counts()
# bins = [0,100,250,500, 1000, np.inf]
# df['CAT_review_count'] = pd.cut(df['review_count'], bins)
# df['CAT_review_count'].value_counts()
# is_open
df.loc[df['is_open']==1,'is_open'] = 'Open'
df.loc[df['is_open']==0,'is_open'] = 'Closed'
# is_rest
df.loc[df['is_rest']==1,'is_rest'] = 'Rest'
df.loc[df['is_rest']==0,'is_rest'] = 'Not Rest'
# Price_Range
df.loc[df['Price_Range']=='1','Price_Range'] = 1.0
df.loc[df['Price_Range']=='2','Price_Range'] = 2.0
df.loc[df['Price_Range']=='3','Price_Range'] = 3.0
df.loc[df['Price_Range']=='4','Price_Range'] = 4.0
# after_covid
df.loc[df['after_covid']==1,'is_rest'] = 'After Covid'
df.loc[df['after_covid']==0,'is_rest'] = 'Before Covid'
# highlight
#df['highlights'] = df['highlights'].str.split(",",expand=False)
# # delivery or takeout
# df.loc[df['delivery or takeout']==1,'delivery or takeout'] = 'delivery or takeout'
# df.loc[df['delivery or takeout']==0,'delivery or takeout'] = 'No delivery or takeout'
# # Grubhub enabled
# df.loc[df['Grubhub enabled']==1,'Grubhub enabled'] = 'Grubhub enabled'
# df.loc[df['Grubhub enabled']==0,'Grubhub enabled'] = 'Grubhub not enabled'
# # Call To Action enabled
# df.loc[df['Call To Action enabled']==1,'Call To Action enabled'] = 'Call To Action enabled'
# df.loc[df['Call To Action enabled']==0,'Call To Action enabled'] = 'Call To Action not enabled'
# # Request a Quote Enabled
# df.loc[df['Request a Quote Enabled']==1,'Request a Quote Enabled'] = 'Request a Quote Enabled'
# df.loc[df['Request a Quote Enabled']==0,'Request a Quote Enabled'] = 'Request a Quote Not Enabled'
# Review_text
df['Review_text']=df['text']
# Restaurant_Covid_Banner
# df['Restaurant_Covid_Banner']=df['Covid Banner']


# In[ ]:


# vars_ = ['business_id','Review_text','Restaurant_Covid_Banner','user_rating','CAT_useful','user_rating','CAT_funny','CAT_cool',
#  'city','state','postal_code', 'latitude', 'longitude','business_rating','CAT_review_count','Price_Range',
# 'is_open','highlights','delivery or takeout','Grubhub enabled','Temporary Closed Until','Virtual Services Offered']


# In[ ]:


# df[df['highlights'].notna()]['highlights']


# In[ ]:


# df.loc[df['gift_cards_during_covid_19'].str.contains('gift_cards_during_covid_19'),
#        'gift_cards_during_covid_19'] = 1


# In[ ]:


# list(df[0:50].columns)


# In[ ]:


# df = df[vars_]


# # 3. Sentiment Analysis

# In[8]:


sid = SentimentIntensityAnalyzer()
df["Review_text_sentiments"] = df["Review_text"].apply(lambda x: sid.polarity_scores(x))
df = pd.concat([df.drop(['Review_text_sentiments'], axis=1), 
                        df['Review_text_sentiments'].apply(pd.Series)], axis=1)


# # 4. LDA Topics

# ### 1. Review_text

# In[9]:


# Convert to list
df['Review_text'] = df['Review_text'].astype(str)
data = df['Review_text'].values.tolist()
# Remove Emails
data = [re.sub(r'\S*@\S*\s?', '', sent) for sent in data]
# Remove new line characters
data = [re.sub(r'\s+', ' ', sent) for sent in data]
# Remove distracting single quotes
data = [re.sub(r"\'", "", sent) for sent in data]
print(data[:1])


# Tokenize

# In[10]:


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
data_words = list(sent_to_words(data))
print(data_words[:1])


# Stemming

# In[11]:


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']): #'NOUN', 'ADJ', 'VERB', 'ADV'
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out


# In[12]:


# Initialize spacy ‘en’ model, keeping only tagger component (for efficiency)
# Run in terminal: python -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])
# Do lemmatization keeping only Noun, Adj, Verb, Adverb
data_lemmatized = lemmatization(data_words, 
                                allowed_postags=['NOUN', 'VERB']) #select noun and verb
print(data_lemmatized[:2])


# Create the Document-Word matrix

# In[13]:


vectorizer = CountVectorizer(analyzer='word',       
                             min_df=10,
# minimum reqd occurences of a word 
                             stop_words='english',             
# remove stop words
                             lowercase=True,                   
# convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  
# num chars > 3
                             # max_features=50000,             
# max number of uniq words    
                            )
data_vectorized = vectorizer.fit_transform(data_lemmatized)


# Build LDA model with sklearn

# In[14]:


# Build LDA Model
lda_model = LatentDirichletAllocation(n_components=10,               # Number of topics
                                      max_iter=3,               
# Max learning iterations
                                      learning_method='online',   
                                      random_state=100,          
# Random state
                                      batch_size=5,            
# n docs in each learning iter
                                      evaluate_every = -1,       
# compute perplexity every n iters, default: Don't
                                      n_jobs = -1,               
# Use all available CPUs
                                     )
lda_output = lda_model.fit_transform(data_vectorized)
print(lda_model)  # Model attributes


# In[15]:


LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
                         evaluate_every=-1, learning_decay=0.7,
                         learning_method='online', learning_offset=10.0,
                         max_doc_update_iter=100, max_iter=10, mean_change_tol=0.001,
                         n_components=10, n_jobs=-1, 
                          #n_topics=20, 
                          perp_tol=0.1,
                         random_state=100, topic_word_prior=None,
                         total_samples=1000000.0, verbose=0)


# In[16]:


# Log Likelyhood: Higher the better
print("Log Likelihood: ", lda_model.score(data_vectorized))
# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity: ", lda_model.perplexity(data_vectorized))
# See model parameters
pprint(lda_model.get_params())


# Use GridSearch to determine the best LDA model.

# In[17]:


# Define Search Param
search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}
# Init the Model
lda = LatentDirichletAllocation(max_iter=5, learning_method='online', 
                                learning_offset=50.,random_state=0)
# Init Grid Search Class
model = GridSearchCV(lda, param_grid=search_params)
# Do the Grid Search
model.fit(data_vectorized)
GridSearchCV(cv=None, error_score='raise',
       estimator=LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
             evaluate_every=-1, learning_decay=0.7, learning_method=None,
             learning_offset=10.0, max_doc_update_iter=100, max_iter=10,
             mean_change_tol=0.001, n_components=10, n_jobs=1,
             #n_topics=None, 
                                           perp_tol=0.1, random_state=None,
             topic_word_prior=None, total_samples=1000000.0, verbose=0),
             iid=True, 
             n_jobs=1,
       param_grid={'n_topics': [10, 15, 20, 25, 30], 'learning_decay': [0.5, 0.7, 0.9]},
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring=None, verbose=0)


# In[18]:


# Best Model
best_lda_model = model.best_estimator_
# Model Parameters
print("Best Model's Params: ", model.best_params_)
# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)
# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))


# Dominant topic

# In[19]:


# Create Document — Topic Matrix
lda_output = best_lda_model.transform(data_vectorized)
# column names
topicnames = ["topic" + str(i) for i in range(best_lda_model.n_components)]
# index names
docnames = ["Doc" + str(i) for i in range(len(data))]
# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)
# Get dominant topic for each document
dominant_topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['dominant_topic'] = dominant_topic
# Styling
def color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)
def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)
# Apply Style
df_document_topics = df_document_topic.head(15).style.applymap(color_green).applymap(make_bold)
df_document_topics


# In[20]:


# Topic-Keyword Matrix
df_topic_keywords = pd.DataFrame(best_lda_model.components_)
# Assign Column and Index
df_topic_keywords.columns = vectorizer.get_feature_names()
df_topic_keywords.index = topicnames
# View
df_topic_keywords.head()


# Get the top 15 keywords each topic

# In[21]:


# Show top n keywords for each topic
def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords
topic_keywords = show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=15)
# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
df_topic_keywords


# # Generate Topics

# In[23]:


Topics = ["Location",
          "Meat",
          "Fast Food",
          "Fish",
          "Menu/Service", 
          "Atmosphere/Staff", 
          "Food flavor", 
          "Order/Waiting time", 
          "Dessert/Coffee", 
          "Staple Food"]
df_topic_keywords["Topics"]=Topics
df_topic_keywords


# In[24]:


# Define function to predict topic for a given text document.
nlp = spacy.load('en', disable=['parser', 'ner'])
def predict_topic(text, nlp=nlp):
    global sent_to_words
    global lemmatization
# Step 1: Clean with simple_preprocess
    mytext_2 = list(sent_to_words(text))
# Step 2: Lemmatize
    mytext_3 = lemmatization(mytext_2, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
# Step 3: Vectorize transform
    mytext_4 = vectorizer.transform(mytext_3)
# Step 4: LDA Transform
    topic_probability_scores = best_lda_model.transform(mytext_4)
    topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), 1:14].values.tolist()
    
    # Step 5: Infer Topic
    infer_topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), -1]
    
    #topic_guess = df_topic_keywords.iloc[np.argmax(topic_probability_scores), Topics]
    return infer_topic, topic, topic_probability_scores
# Predict the topic
mytext = ["Very Useful in diabetes age 30. I need control sugar. thanks Good deal"]
infer_topic, topic, prob_scores = predict_topic(text = mytext)
print(topic)
print(infer_topic)


# Predict topics of our reviews in the original dataset:

# In[33]:


def apply_predict_topic(text):
    text = [text]
    infer_topic, topic, prob_scores = predict_topic(text = text)
    return(infer_topic)
df["Review_text_90Topic_key_word"]= df['Review_text'].apply(apply_predict_topic)
df.head()


# In[37]:


df = df.rename(columns = {'Review_text_90Topic_key_word':'Review_text_Topic_key_word'})


# In[38]:


df['Review_text_Topic_key_word'].value_counts()


# In[39]:


df.to_csv('04192021_output.csv')


# #### How to cluster documents that share similar topics and plot?

# In[ ]:


# Construct the k-means clusters
from sklearn.cluster import KMeans
clusters = KMeans(n_clusters=15, random_state=100).fit_predict(lda_output)
# Build the Singular Value Decomposition(SVD) model
svd_model = TruncatedSVD(n_components=2)  # 2 components
lda_output_svd = svd_model.fit_transform(lda_output)
# X and Y axes of the plot using SVD decomposition
x = lda_output_svd[:, 0]
y = lda_output_svd[:, 1]
# Weights for the 15 columns of lda_output, for each component
print("Component's weights: \n", np.round(svd_model.components_, 2))
# Percentage of total information in 'lda_output' explained by the two components
print("Perc of Variance Explained: \n", np.round(svd_model.explained_variance_ratio_, 2))


# In[ ]:


# Plot
plt.figure(figsize=(12, 12))
plt.scatter(x, y, c=clusters)
plt.xlabel('Component 2')
plt.xlabel('Component 1')
plt.title("Segregation of Topic Clusters", )


# In[ ]:





# In[ ]:





# In[ ]:




