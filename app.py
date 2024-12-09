from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load datasets
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Merge datasets
data = pd.merge(ratings, movies, on='movieId')

# Feature Engineering
data['genres'] = data['genres'].str.split('|')
genres = data['genres'].apply(lambda x: pd.Series(1, index=x)).fillna(0)
data = pd.concat([data, genres], axis=1)
data.drop(columns='genres', inplace=True)

# Standarize numeric features
scaler = StandardScaler()
data['rating_scaled'] = scaler.fit_transform(data[['rating']])

# Prepare features for similarity computation
features = genres.columns.tolist() + ['rating_scaled']
movie_features = data.groupby('movieId')[features].mean()

# Compute cosine similarity
similarity_matrix = cosine_similarity(movie_features)


def recommend_movies(favorite_movie_ids, n_recommendations=5):
    similarity_scores = np.zeros(similarity_matrix.shape[0])
    
    for movie_id in favorite_movie_ids:
        if movie_id in movie_features.index:
            movie_idx = movie_features.index.tolist().index(movie_id)
            similarity_scores += similarity_matrix[movie_idx]
    
    sorted_indices = np.argsort(similarity_scores)[::-1]
    recommendations = [
        movie_features.index[i] for i in sorted_indices if movie_features.index[i] not in favorite_movie_ids
    ]
    top_recommendations = recommendations[:n_recommendations]
    
    return [
        {
            "movieId": mid,
            "title": movies[movies['movieId'] == mid]['title'].values[0],
            "genres": movies[movies['movieId'] == mid]['genres'].values[0],
            "rating": round(ratings[ratings['movieId'] == mid]['rating'].mean(), 2)
        }
        for mid in top_recommendations
    ]


@app.route('/')
def home():
    return render_template('index.html', movies=None)

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']  # Ambil input pengguna
    matched_movies = movies[movies['title'].str.contains(query, case=False, na=False)]
    
    if matched_movies.empty:
        return render_template('index.html', movies=None, error="No movies found.", query=query)
    
    matched_ids = matched_movies['movieId'].tolist()
    if len(matched_ids) > 1:
        movie_list = [
            {
                "title": movies[movies['movieId'] == mid]['title'].values[0],
                "id": mid
            }
            for mid in matched_ids
        ]
        return render_template('index.html', movies=movie_list, multiple=True, query=query)

    favorite_movie_ids = matched_ids[:1]
    recommendations = recommend_movies(favorite_movie_ids, n_recommendations=5)
    return render_template('index.html', movies=recommendations, query=query, multiple=False)



@app.route('/recommend/<int:movie_id>')
def recommend(movie_id):
    # Ambil data film berdasarkan movie_id
    selected_movie = movies[movies['movieId'] == movie_id]
    if selected_movie.empty:
        return render_template('index.html', movies=None, error="Movie not found.", query=None)

    # Dapatkan judul film yang dipilih
    selected_title = selected_movie['title'].values[0]

    # Berikan rekomendasi berdasarkan film yang dipilih
    recommendations = recommend_movies([movie_id], n_recommendations=5)
    
    # Kirimkan rekomendasi dan judul yang dipilih kembali ke template
    return render_template('index.html', movies=recommendations, query=selected_title)



if __name__ == '__main__':
    app.run(debug=True)
