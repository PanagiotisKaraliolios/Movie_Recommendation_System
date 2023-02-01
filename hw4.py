# Create a movies' recommendation system using a collaborative filtering algorithm.
# The program will first read in the data from the MovieLens 100k dataset.
# It will then use the data to build a model that predicts the rating of a user
# for a given movie.

# import the necessary packages
import csv
import operator
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from tqdm.auto import tqdm


# define a function that reads the ratings from a csv
def read_ratings(filename):
    """
    Reads the ratings from a csv and returns a list of tuples
    (user_id, movie_id, rating, timestamp)

    """
    column_names = ['userId', 'movieId', 'rating', 'timestamp']
    ratings = pd.read_csv(filename, sep=',', names=column_names,
                          encoding='utf-8', quoting=csv.QUOTE_ALL, skipinitialspace=True)
    ratings = ratings.dropna()
    # drop first row
    ratings = ratings.drop(ratings.index[0])
    # change rating to float type
    ratings['rating'] = ratings['rating'].astype(float)

    # drop the timestamp column
    ratings = ratings.drop(columns=['timestamp'])

    return ratings


# define a function that calculates the jaccard similarity between two sets
def jaccard_similarity(set1, set2):
    """
    Calculates the jaccard similarity between two sets

    """

    # calculate the jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    jaccard_sim = intersection / union

    return jaccard_sim


# define a function that calculates the adjusted cosine similarity between two sets of ratings
def adjusted_cosine_similarity(set1, set2):
    """
    Calculates the adjusted cosine similarity between two sets of ratings

    """
    # calculate the adjusted cosine similarity
    numerator = np.dot(set1, set2)
    denominator = np.sqrt(np.sum(np.square(set1))) * \
        np.sqrt(np.sum(np.square(set2)))
    if denominator == 0:
        return 0
    adjusted_cosine = numerator / denominator

    return adjusted_cosine


# define a function that returns the movies similarities dictionary and the users similarities dictionary
def get_similarities(train_data, model_choice):
    """
    Returns the movies similarities dictionary and the users similarities dictionary
    :param train_data:
    :param model_choice:
    :return: movies_similarities, users_similarities
    """
    train_data_pivot = train_data.pivot(
        index='userId', columns='movieId', values='rating')

    # normalize each user's ratings
    train_data_pivot = train_data_pivot.apply(lambda x: (x - np.nanmean(x)))

    # fill in the missing values with 0
    train_data_pivot = train_data_pivot.fillna(0)

    movie_similarity = {}
    # if the model choice is 1, calculate the similarity between the training data movies with Jaccard similarity
    if model_choice == 1:
        # create a dictionary that maps the movieId on the training set to a set of users that have rated it
        movie_user_dict = {}

        for index, row in train_data.iterrows():
            movie = row['movieId']
            user = row['userId']
            if movie in movie_user_dict:
                movie_user_dict[movie].add(user)
            else:
                movie_user_dict[movie] = {user}

        # print an empty line
        print()

        # for each movie in the movie_to_users dictionary, calculate the jaccard similarity
        # between the movie and all the other movies
        for movie in tqdm(movie_user_dict, desc='Calculating movie similarities', unit=' movies'):
            movie_similarity[movie] = {}
            for other_movie_id in movie_user_dict:
                if movie != other_movie_id:
                    movie_similarity[movie][other_movie_id] = jaccard_similarity(movie_user_dict[movie],
                                                                                 movie_user_dict[other_movie_id])

        # sort each movie similarities in movie_similarity dictionary by the values in descending order
        for movie in movie_similarity:
            movie_similarity[movie] = dict(
                sorted(movie_similarity[movie].items(), key=operator.itemgetter(1), reverse=True))

    # else if the model choice is 2, calculate the similarity between the training data movies with adjusted cosine similarity
    elif model_choice == 2:

        movie_ratings_dict = {}

        for movieId, values in train_data_pivot.iteritems():
            movie_ratings_dict[movieId] = np.array(values)

        # print an empty line
        print()

        # for each movie in the movie_rating_dict, calculate the adjusted cosine similarity between the movie and all the other movies
        for movie in tqdm(movie_ratings_dict, desc='Calculating movie similarities', unit=' movies'):
            movie_similarity[movie] = {}
            for other_movie_id in movie_ratings_dict:
                if movie != other_movie_id:
                    # calculate the adjusted cosine similarity
                    movie_ratings = movie_ratings_dict[movie]
                    other_movie_ratings = movie_ratings_dict[other_movie_id]

                    movie_similarity[movie][other_movie_id] = adjusted_cosine_similarity(movie_ratings,
                                                                                         other_movie_ratings)

        # sort each movie similarities in movie_similarity dictionary by the values in descending order
        for movie in movie_similarity:
            movie_similarity[movie] = dict(
                sorted(movie_similarity[movie].items(), key=operator.itemgetter(1), reverse=True))

    user_similarity = {}
    # calculate the adjusted cosine similarity between the users in the training data
    # create a dictionary that maps userId to its normalized ratings
    user_rating_dict = {}
    # each row in the training data pivot table is a user
    for user_id, values in train_data_pivot.iterrows():
        user_rating_dict[user_id] = np.array(values)

    # print an empty line
    print()

    # for each user in the user_rating_dict, calculate the adjusted cosine similarity between the user and all the other users
    for user_id in tqdm(user_rating_dict, desc='Calculating user similarities', unit=' users'):
        user_similarity[user_id] = {}
        for other_user_id in user_rating_dict:
            if user_id != other_user_id:
                user_ratings = user_rating_dict[user_id]
                other_user_ratings = user_rating_dict[other_user_id]

                # calculate the adjusted cosine similarity
                user_similarity[user_id][other_user_id] = adjusted_cosine_similarity(
                    user_ratings, other_user_ratings)

    # sort each user similarities in user_similarity dictionary by the values in descending order
    for user_id in user_similarity:
        user_similarity[user_id] = dict(
            sorted(user_similarity[user_id].items(), key=operator.itemgetter(1), reverse=True))

    return movie_similarity, user_similarity


if __name__ == '__main__':

    # ask for the user to specify the K value and the percentage of the data to be used for training.
    # If K value is below 1, print an error message and prompt the user to enter a valid value.
    while (K := int(input('Enter the K value: '))) < 1:
        print('K value must be greater than 0')

    # if the percentage is above 90%, print an error message and prompt the user to enter a new percentage
    while (train_data_percentage := float(
            input(
                'Please specify the percentage of the data to be used for training. It must be below 90%(p.e. 0.8 = 80%): '))) > 0.9:
        print('The percentage must be below 90%(0.1 - 0.9). Please try again.')

    # ask the user to specify which model to use. 1 = S1 with Jaccard similarity, and S2 with adjusted cosine similarity, 2 = Both S1 and S2 with adjusted cosine similarity
    while (model_choice := int(
        input(
            'Please specify the model to use. 1 = S1 with Jaccard similarity, and S2 with adjusted cosine similarity, 2 = Both S1 and S2 with adjusted cosine similarity: '))) not in [
            1, 2]:
        print('The model choice must be 1 or 2')

    # read the ratings
    ratings = read_ratings("ratings.csv")

    # count the number of ratings for each movieId
    movie_counts = ratings.groupby('movieId').size()
    # sort the movie_counts in descending order
    movie_counts = movie_counts.sort_values(ascending=False)

    # in movie_counts, drop the movies that have less than 5 ratings
    movie_counts = movie_counts[movie_counts >= 5]

    # from the ratings, keep only the ratings for the movies that are in movie_counts
    ratings = ratings[ratings['movieId'].isin(movie_counts.index)]

    # split the ratings dataframe into training and test data
    # get the random 10 % of the data as test data and the train_data_percentage % of the data as training data
    test_data = ratings.sample(frac=0.1)

    # count how many ratings will be used for training
    train_data_count = int(train_data_percentage * len(ratings))

    # get the train_data_count as random training data
    train_data = ratings.drop(test_data.index)
    train_data = train_data.sample(train_data_count)

    # get the similarities dictionaries
    movie_similarity, user_similarity = get_similarities(
        train_data, model_choice)

    # Start Phase S1

    # print an empty line
    print()

    # for each entry in the test_data
    for index, row in tqdm(test_data.iterrows(), desc='Phase S1', unit=' entries', total=len(test_data)):
        # get the userId, movieId and rating
        user_id = row['userId']
        movie_id = row['movieId']
        rating = row['rating']

        # if the movieId is not in the movie_similarity dictionary, drop entry and continue to the next
        if movie_id not in movie_similarity:
            test_data.drop(index, inplace=True)
            continue
        # get the similar movies for the movie_id
        similar_movies = movie_similarity[movie_id]

        # get all the ratings that match the user_id and the movies in similar_movies
        similar_movies_ratings = train_data[
            (train_data['userId'] == user_id) & (train_data['movieId'].isin(similar_movies))]

        # create a dataframe with the similar_movies_ratings and the similarity values
        similar_movies_ratings = similar_movies_ratings.assign(
            similarity=similar_movies_ratings['movieId'].map(lambda x: movie_similarity[movie_id][x]))

        # drop the movies that have similarity 0 or less
        similar_movies_ratings = similar_movies_ratings[similar_movies_ratings['similarity'] > 0]

        # sort the similar_movies_ratings by the similarity values in descending order and keep the top K similar movies
        similar_movies_ratings = similar_movies_ratings.sort_values(
            by='similarity', ascending=False).head(K)

        # if the similar_movies_ratings is empty, then skip this entry
        if len(similar_movies_ratings) != 0:

            # multiply each rating by the similarity value
            similar_movies_ratings['rating'] = similar_movies_ratings['rating'] * \
                similar_movies_ratings['similarity']

            # sum the ratings
            sum_similar_movies_ratings = similar_movies_ratings['rating'].sum()

            # sum the similarities
            sum_similar_movies_ratings_similarity = similar_movies_ratings['similarity'].sum(
            )

            # calculate the predicted rating
            predicted_rating = sum_similar_movies_ratings / \
                sum_similar_movies_ratings_similarity

            # if the predicted rating is less than 2.5, remove the entry from the test_data
            if predicted_rating < 2.5:
                test_data = test_data.drop(index)

        else:
            test_data = test_data.drop(index)

    # End Phase S1

    # Start Phase S2

    # create a dataframe to store the predicted ratings
    predicted_ratings = pd.DataFrame(columns=['userId', 'movieId', 'rating'])

    # print an empty line
    print()

    # for each entry in the test_data
    for index, row in tqdm(test_data.iterrows(), desc='Phase S2', unit=' entries', total=len(test_data)):
        # get the userId, movieId and rating
        user_id = row['userId']
        movie_id = row['movieId']
        rating = row['rating']

        # get the similar users for the user_id
        similar_users = user_similarity[user_id]

        # get all the ratings that match the movie_id and the users in similar_users
        similar_users_ratings = train_data[
            (train_data['userId'].isin(similar_users)) & (train_data['movieId'] == movie_id)]

        # create a dataframe with the similar_users_ratings and the similarity values
        similar_users_ratings = similar_users_ratings.assign(
            similarity=similar_users_ratings['userId'].map(lambda x: user_similarity[user_id][x]))

        # drop the users that have similarity 0 or less
        similar_users_ratings = similar_users_ratings[similar_users_ratings['similarity'] > 0]

        # sort the similar_users_ratings by the similarity values in descending order and keep the top K similar users
        similar_users_ratings = similar_users_ratings.sort_values(
            by='similarity', ascending=False).head(K)

        # if the similar_users_ratings is empty, then skip this entry
        if len(similar_users_ratings) != 0:

            # multiply each rating by the similarity value
            similar_users_ratings['rating'] = similar_users_ratings['rating'] * \
                similar_users_ratings['similarity']

            # sum the ratings
            sum_similar_users_ratings = similar_users_ratings['rating'].sum()

            # sum the similarities
            sum_similar_users_ratings_similarity = similar_users_ratings['similarity'].sum(
            )

            # calculate the predicted rating
            predicted_rating = sum_similar_users_ratings / \
                sum_similar_users_ratings_similarity

            # update the predicted_ratings dataframe with the predicted rating
            # Create a new dataframe with the predicted rating
            d = {'userId': user_id, 'movieId': movie_id,
                 'rating': predicted_rating}
            S1 = pd.Series(data=d)
            predicted_ratings = pd.concat(
                [predicted_ratings, S1.to_frame().T], ignore_index=True)

        else:
            # update the predicted_ratings dataframe with NaN value as rating
            # Create a new dataframe with the predicted rating
            d = {'userId': user_id, 'movieId': movie_id, 'rating': np.nan}
            S1 = pd.Series(data=d)
            predicted_ratings = pd.concat(
                [predicted_ratings, S1.to_frame().T], ignore_index=True)

    # End Phase S2

    # from the predicted_ratings dataframe, drop the rows where the rating is NaN
    predicted_ratings = predicted_ratings.dropna()

    # in test_data, reset the index
    test_data = test_data.reset_index(drop=True)

    # drop the rows whose index is not in the predicted_ratings dataframe
    test_data = test_data.drop(
        test_data[test_data.index.isin(predicted_ratings.index) == False].index)

    # Calculate the Mean Absolute Error
    mae = mean_absolute_error(test_data['rating'], predicted_ratings['rating'])
    print('Mean Absolute Error: ', mae)

    # Calculate the Recall and Precision
    # If the predicted rating is greater than 3.5, then it is a positive prediction
    predicted_ratings['positivity'] = predicted_ratings['rating'].apply(
        lambda x: 1 if x > 3.5 else 0)

    # On the test data, if the rating is greater than 3.5, then it is a positive prediction
    test_data['positivity'] = test_data['rating'].apply(
        lambda x: 1 if x > 3.5 else 0)

    # Create a dataframe to store the two positivity columns
    positivity = pd.DataFrame(columns=['predicted', 'actual'])
    positivity['predicted'] = predicted_ratings['positivity']
    positivity['actual'] = test_data['positivity']

    # Count the true positives, true negatives, false positives and false negatives
    true_positives = positivity[(positivity['actual'] == 1) & (
        positivity['predicted'] == 1)].shape[0]
    true_negatives = positivity[(positivity['actual'] == 0) & (
        positivity['predicted'] == 0)].shape[0]
    false_positives = positivity[(positivity['actual'] == 0) & (
        positivity['predicted'] == 1)].shape[0]
    false_negatives = positivity[(positivity['actual'] == 1) & (
        positivity['predicted'] == 0)].shape[0]

    # Calculate the Recall
    recall = true_positives / (true_positives + false_negatives)
    print('Recall: ', recall)

    # Calculate the Precision
    precision = true_positives / (true_positives + false_positives)
    print('Precision: ', precision)

    print("###### End of the program ######")
# end of the program
