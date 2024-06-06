# python-project-2
restaurant recommendation system

a basic restaurant recommendation system implemented in Python using concepts of machine learning. This program uses a simple collaborative filtering approach to recommend restaurants based on user ratings.

python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sample user ratings data
ratings = {
    'user1': {'restaurant1': 4, 'restaurant2': 5, 'restaurant3': 3, 'restaurant4': 2},
    'user2': {'restaurant1': 3, 'restaurant2': 4, 'restaurant3': 2, 'restaurant4': 5},
    'user3': {'restaurant1': 5, 'restaurant2': 2, 'restaurant3': 4, 'restaurant4': 3},
    'user4': {'restaurant1': 2, 'restaurant2': 3, 'restaurant3': 5, 'restaurant4': 4}
}

# Convert ratings to a numpy array
ratings_matrix = np.array([[ratings[user][restaurant] for restaurant in ratings[user]] for user in ratings])

# Calculate cosine similarity matrix
similarity_matrix = cosine_similarity(ratings_matrix)

# Function to get top n similar users
def get_top_similar_users(user_id, n=2):
    user_index = list(ratings.keys()).index(user_id)
    sim_users = similarity_matrix[user_index]
    top_similar_users_indices = np.argsort(sim_users)[-n-1:-1][::-1]  # Exclude self-similarity
    return [list(ratings.keys())[i] for i in top_similar_users_indices]

# Function to recommend restaurants to a user
def recommend_restaurants(user_id, n=2):
    similar_users = get_top_similar_users(user_id)
    user_index = list(ratings.keys()).index(user_id)
    recommended_restaurants = {}
    
    for restaurant in ratings[similar_users[0]]:
        if restaurant not in ratings[user_id]:
            total_rating = 0
            total_similarity = 0
            for similar_user in similar_users:
                if restaurant in ratings[similar_user]:
                    total_rating += ratings[similar_user][restaurant] * similarity_matrix[user_index][list(ratings.keys()).index(similar_user)]
                    total_similarity += similarity_matrix[user_index][list(ratings.keys()).index(similar_user)]
            recommended_restaurants[restaurant] = total_rating / total_similarity if total_similarity != 0 else 0
            
    return dict(sorted(recommended_restaurants.items(), key=lambda x: x[1], reverse=True)[:n])

# Test the recommendation system
user_id = 'user1'
recommended_restaurants = recommend_restaurants(user_id)
print(f"Recommended restaurants for {user_id}: {recommended_restaurants}")


This code first defines a sample user ratings dataset, then converts it into a numpy array. It calculates the cosine similarity matrix between users based on their ratings. The get_top_similar_users function returns the top similar users to a given user, and the recommend_restaurants function recommends restaurants to the user based on their similarity with other users' ratings. Finally, a test case is provided to demonstrate how to use the recommendation system.
