import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from openai import OpenAI
import json
import re
import pdb


def load_node_csv(path, index_col):
    """Loads csv containing node information

    Args:
        path (str): path to csv file
        index_col (str): column name of index column

    Returns:
        dict: mapping of csv row to node id
    """
    df = pd.read_csv(path, index_col=index_col)
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    return mapping


def load_edge_csv(
    path,
    src_index_col,
    src_mapping,
    dst_index_col,
    dst_mapping,
    link_index_col,
    rating_threshold=4,
):
    """Loads csv containing edges between users and items

    Args:
        path (str): path to csv file
        src_index_col (str): column name of users
        src_mapping (dict): mapping between row number and user id
        dst_index_col (str): column name of items
        dst_mapping (dict): mapping between row number and item id
        link_index_col (str): column name of user item interaction
        rating_threshold (int, optional): Threshold to determine positivity of edge. Defaults to 4.

    Returns:
        torch.Tensor: 2 by N matrix containing the node ids of N user-item edges
    """
    df = pd.read_csv(path)
    edge_index = None
    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_attr = (
        torch.from_numpy(df[link_index_col].values).view(-1, 1).to(torch.long)
        >= rating_threshold
    )

    edge_index = [[], []]
    for i in range(edge_attr.shape[0]):
        if edge_attr[i]:
            edge_index[0].append(src[i])
            edge_index[1].append(dst[i])

    return torch.tensor(edge_index)


def load_data():
    movie_path = os.path.join(os.path.dirname(__file__), "ml-latest-small/movies.csv")
    rating_path = os.path.join(os.path.dirname(__file__), "ml-latest-small/ratings.csv")
    user_mapping = load_node_csv(rating_path, index_col="userId")
    movie_mapping = load_node_csv(movie_path, index_col="movieId")
    edge_index = load_edge_csv(
        rating_path,
        src_index_col="userId",
        src_mapping=user_mapping,
        dst_index_col="movieId",
        dst_mapping=movie_mapping,
        link_index_col="rating",
        rating_threshold=4,
    )
    num_interactions = edge_index.shape[1]
    train_indices, test_indices = train_test_split(
        np.arange(num_interactions), test_size=0.2
    )
    val_indices, test_indices = train_test_split(
        test_indices, test_size=0.5, random_state=1
    )
    val_edge_index = edge_index[:, val_indices]
    test_edge_index = edge_index[:, test_indices]
    train_edge_index = edge_index[:, train_indices]

    return (
        torch.tensor(edge_index),
        user_mapping,
        movie_mapping,
        test_edge_index,
        val_edge_index,
        train_edge_index,
    )


def get_user_positive_items(edge_index):
    """Generates dictionary of positive items for each user

    Args:
        edge_index (torch.Tensor): 2 by N list of edges

    Returns:
        dict: dictionary of positive items for each user
    """
    user_pos_items = {}
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in user_pos_items:
            user_pos_items[user] = []
        user_pos_items[user].append(item)
    return user_pos_items


def get_movie_title_and_genre():
    movie_path = os.path.join(os.path.dirname(__file__), "ml-latest-small/movies.csv")
    df = pd.read_csv(movie_path)
    movieid_title = df.set_index("movieId")["title"].to_dict()
    movieid_genres = df.set_index("movieId")["genres"].to_dict()
    return movieid_title, movieid_genres


def generate_text(prompt, model, openai_api_key):
    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=2048,
        top_p=1,
        messages=[
            {"role": "system", "content": "You are a hardworking assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    message_content = response.choices[0].message.content.strip()

    return message_content


def extract_movies(json_string):
    if not json_string.strip():
        raise ValueError("Empty JSON string")

    try:
        # Parse the JSON string
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {e}")

    # Extract the values
    movies_removed = data.get("movies_removed", [])
    movies_reco = data.get("movies_reco", [])

    return movies_removed, movies_reco

def calculate_recall_precision(gr, recommended_movies):
      # print("ground truth", gr)
      # print("recommended movies", recommended_movies)
      liked_set = set(gr)
      recommended_set = set(recommended_movies)
      true_positives = len(liked_set.intersection(recommended_set))
      false_negatives = len(liked_set.difference(recommended_set))
      false_positives = len(recommended_set.difference(liked_set))
      recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
      precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
      total_predictions = len(recommended_movies)
      return recall, precision
