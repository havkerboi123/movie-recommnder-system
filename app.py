import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

# Loading data
movies = pd.read_csv('/Users/muhammadahmed/Downloads/archive-4/tmdb_5000_movies.csv')
credits = pd.read_csv('/Users/muhammadahmed/Downloads/archive-4/tmdb_5000_credits.csv')

# Merging datasets
movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'vote_average']]

# Data preprocessing for content-based filtering
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 

movies.dropna(inplace=True)
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: x[0:3])

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 

movies['crew'] = movies['crew'].apply(fetch_director)

def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
new['tags'] = new['tags'].apply(lambda x: " ".join(x))

# Content-based filtering
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new['tags']).toarray()
similarity_content = cosine_similarity(vector)

# Collaborative filtering (SVD)
reader = Reader(rating_scale=(0, 10))
data = Dataset.load_from_df(movies[['movie_id', 'title', 'vote_average']], reader)
trainset = data.build_full_trainset()
svd = SVD()


svd.fit(trainset)


X = movies[['movie_id', 'vote_average']]
y = movies['vote_average']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Streamlit app
st.title("Movie Recommender System")


movie_input = st.sidebar.selectbox("Select a movie:", movies['title'])


st.header("Content-Based Recommendations:")
st.write(f"Top 10 recommendations for {movie_input} based on content:")

movie_index = movies.loc[movies['title'] == movie_input].index
if not movie_index.empty:
    movie_index = movie_index[0]
    sim_scores = list(enumerate(similarity_content[movie_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Exclude the input movie itself
    movie_indices = [i[0] for i in sim_scores]
    content_recommendations = movies[['title', 'vote_average']].iloc[movie_indices]
    st.table(content_recommendations)
else:
    st.write(f"Movie '{movie_input}' not found in the dataset.")


st.header("Collaborative Filtering (SVD) Recommendations:")
st.write(f"Top 10 recommendations for {movie_input} based on collaborative filtering:")

collab_recommendations = pd.DataFrame({
    'title': movies['title'].head(trainset.n_items),
    'collab_rating': [svd.predict(movie_id, 1).est for movie_id in range(1, trainset.n_items + 1)]
})

collab_recommendations = collab_recommendations.sort_values(by='collab_rating', ascending=False).head(10)

# Join the recommendations with the movies DataFrame based on movie titles
collab_recommendations = collab_recommendations.merge(movies[['title', 'vote_average']], left_on='title', right_on='title')

st.table(collab_recommendations[['title', 'vote_average']])

# Linear Regression recommendations
st.header("Linear Regression Recommendations:")
st.write(f"Top 10 recommendations for {movie_input} based on linear regression:")
lr_prediction = lr_model.predict([[movie_index, 0]])[0]
lr_recommendations = pd.DataFrame({'title': movies['title'], 'vote_average': [lr_prediction] * len(movies)})
lr_recommendations = lr_recommendations.sort_values(by='vote_average', ascending=False).head(10)
st.table(lr_recommendations[['title', 'vote_average']])
