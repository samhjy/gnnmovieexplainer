import os
import uvicorn
import torch

from fastapi import FastAPI, Form, Request, status
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import uuid
import utils
import json
from dotenv import load_dotenv
from openai import OpenAI

from model import LightGCN

import sys
print(sys.path)

ml_models = {}
ml_data = {}
openai_api = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load env variable
    load_dotenv()

    # load lightGCN model from path
    model_path = os.path.join(os.path.dirname(__file__), "models/lightgcn2.pt")
    ml_models["movie_recommender"] = torch.load(model_path)
    ml_models["movie_recommender"].eval()

    # load data for evaluation
    (
        edge_index,
        user_mapping,
        movie_mapping,
        test_edge_index,
        val_edge_index,
        train_edge_index,
    ) = utils.load_data()
    movieid_title, movieid_genre = utils.get_movie_title_and_genre()
    ml_data["test_edge_index"] = test_edge_index
    ml_data["train_edge_index"] = train_edge_index
    ml_data["val_edge_index"] = val_edge_index
    ml_data["edge_index"] = edge_index
    ml_data["user_mapping"] = user_mapping
    ml_data["movie_mapping"] = movie_mapping
    ml_data["movieid_title"] = movieid_title
    ml_data["movieid_genre"] = movieid_genre
    yield


app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

templates_directory = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_directory)

# Mount the static directory
static_directory = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_directory), name="static")


@app.get("/healthz")
async def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "chat_title": "MOVIE RECOMMENDATIONS WITH GNN EXPLAINER",
            "welcome_message": "Provide a list of your favourite movies and I will recommend you some similar movies",
        },
    )


@app.post("/chat")
async def chat(request: Request, message: str = Form(...)) -> HTMLResponse:
    user_id = 1  # Example user_id, replace with actual logic to get user_id
    num_recs = 2  # Example number of recommendations, replace with actual logic to get num_recs

    bot_response = await movie_recommendation(request, user_id, num_recs)
    bot_response_content = bot_response.body.decode("utf-8")
    bot_response_json = json.loads(bot_response_content)

    users_movies = ", ".join(
        [movie["title"] for movie in bot_response_json["users_movies"]]
    )
    recommended_movies = ", ".join(
        [movie["title"] for movie in bot_response_json["recommended_movies"]]
    )

    bot_message = f"Based on the movies you like [{users_movies}]\n\nHere are your recommendations:\n[{recommended_movies}]"

    # get the explanation based on the recommended movies
    explain_res = await explain_results(bot_message)
    explain_res_str = json.dumps(explain_res.body.decode("utf-8"))
    bot_response_html = markdown2.markdown(bot_message, safe_mode="escape")
    message_id = str(uuid.uuid4())

    response_html = templates.TemplateResponse(
        "message.html",
        {
            "request": request,
            "bot_response_html": bot_response_html,
            "message_id": message_id,
            "explain_result": explain_res_str,
        },
    )

    return response_html


@app.post("/lightgcn", status_code=status.HTTP_201_CREATED)
async def movie_recommendation(request: Request, user_id: int, num_recs: int):
    user = ml_data["user_mapping"][user_id]
    e_u = ml_models["movie_recommender"].users_emb.weight[user]
    scores = ml_models["movie_recommender"].items_emb.weight @ e_u
    user_pos_items = utils.get_user_positive_items(ml_data["edge_index"])
    values, indices = torch.topk(scores, k=len(user_pos_items[user]) + num_recs)

    # Get the movies that the user likes
    movies = [index.cpu().item() for index in indices if index in user_pos_items[user]][
        :10
    ]
    movie_ids = [
        list(ml_data["movie_mapping"].keys())[
            list(ml_data["movie_mapping"].values()).index(movie)
        ]
        for movie in movies
    ]
    titles = [ml_data["movieid_title"][id] for id in movie_ids]
    genres = [ml_data["movieid_genre"][id] for id in movie_ids]

    # Get the movies that based on model prediction
    inference_movies = [
        index.cpu().item() for index in indices if index not in user_pos_items[user]
    ][:num_recs]
    inference_movie_ids = [
        list(ml_data["movie_mapping"].keys())[
            list(ml_data["movie_mapping"].values()).index(movie)
        ]
        for movie in inference_movies
    ]
    inference_titles = [ml_data["movieid_title"][id] for id in inference_movie_ids]
    inference_genres = [ml_data["movieid_genre"][id] for id in inference_movie_ids]

    users_movies = [
        {"title": titles[i], "genres": genres[i]} for i in range(len(titles))
    ]
    recommended_movies = [
        {"title": inference_titles[i], "genres": inference_genres[i]}
        for i in range(num_recs)
    ]
    return JSONResponse(
        content={"recommended_movies": recommended_movies, "users_movies": users_movies}
    )


@app.post("/explain", status_code=status.HTTP_201_CREATED)
async def explain_results(predicted_data: str = Form(...)):
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=2048,
        top_p=0.95,
        messages=[
            {
                "role": "system",
                "content": """
                    The following is a output generated with a trained lightGCN model based on what the users like. 
                    Can you provide an explanation why the model recommended each movie?
                    Please output strictly in this JSON format without any additional text
                    """,
            },
            {"role": "user", "content": f"{predicted_data}"},
        ],
    )
    message_content = response.choices[0].message.content

    return JSONResponse(content={"explanation": message_content})


# @app.get("/precision", status_code=status.HTTP_200_OK)
# async def evaluation(request: Request, user_id: int):
#     OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
#     topK = 20
#     user = ml_data["user_mapping"][user_id]
#     e_u = ml_models["movie_recommender"].users_emb.weight[user]
#     scores = ml_models["movie_recommender"].items_emb.weight @ e_u
#     user_pos_items = utils.get_user_positive_items(ml_data["edge_index"])
#     user_train_pos_items = utils.get_user_positive_items(ml_data["train_edge_index"])
#     user_test_pos_items = utils.get_user_positive_items(ml_data["test_edge_index"])
#     user_val_pos_items = utils.get_user_positive_items(ml_data["val_edge_index"])
#     values, indices = torch.topk(scores, k=len(user_pos_items[user]) + topK)
#
#     hrated_movies = [
#         index.cpu().item() for index in indices if index in user_train_pos_items[user]
#     ]  # movies that were rated highly by the user and model knows
#     hrated_movie_ids = [
#         list(ml_data["movie_mapping"].keys())[
#             list(ml_data["movie_mapping"].values()).index(movie)
#         ]
#         for movie in hrated_movies
#     ]  # get the movie ids
#     hrated_titles = [ml_data["movieid_title"][id] for id in hrated_movie_ids]
#     hrated_genres = [ml_data["movieid_genre"][id] for id in hrated_movie_ids]
#
#     user_pos_items = utils.get_user_positive_items(ml_data["edge_index"])
#     # user_test_pos_items = utils.get_user_positive_items(ml_data["test_edge_index"])
#
#     rec_ids_real = [index.cpu().item() for index in indices if index not in user_train_pos_items[user]]
#
#     topk_movies_rec = rec_ids_real[:topK]  # top movies recommended by model
#
#     rec_map = [list(ml_data["movie_mapping"].keys())[list(ml_data["movie_mapping"].values()).index(movie)] for movie in
#                topk_movies_rec]  # movie ids recommended by model
#
#     rec_titles = [ml_data["movieid_title"][id] for id in rec_map]
#     rec_genres = [ml_data["movieid_genre"][id] for id in rec_map]
#     gr_user_pos_items = utils.get_user_positive_items(ml_data["edge_index"])
#     ground_truth = user_test_pos_items[user]  # ground truth for the user, all highly rated movies by the user
#     titles_liked_json = json.dumps(hrated_titles)
#     titles_rec_json = json.dumps(rec_titles)
#     genres_liked_json = json.dumps(hrated_genres)
#     print("recommended titles by LightGCN Model", titles_rec_json)
#     print("liked movies", titles_liked_json)
#
#     prompt = "Based on the user's history , can you generate a user profile and provide details of their {age: , gender, genres liked:, genres dislike:, favourite directors:, country;}. where information is not provided, make an educated guess."
#     prompt += "can you recommend movies that you predict the user will rate highly by profiling them based on the movies they like and other users likes? Base these recommendations on your learnt knowledge of movies similar users are liking and online movies rating sources. return it as a list in this format movies_reco = [" ", " "]. ensure there are quotes and commas."
#     prompt += "this is the movies the user liked"
#     prompt += titles_liked_json
#     prompt += "this is the genres of movies the user liked"
#     prompt += genres_liked_json
#     prompt += "Please remove any movies from the model's predictions that you think does not fit the user profile and preferences, from those towards the end of the list as the recommendations have been ranked by the model already, it is perfoming with an recall of 10%. return it as a list in this format movies_removed = [" ", " "]. ensure there are quotes and commas."
#     prompt += "this is what the model predicted"
#     prompt += titles_rec_json
#     prompt += """
#                 Please output strictly in this JSON format without any additional text:
#
#                 {
#                     "movies_removed": [/* movies removed */],
#                     "movies_reco": [/* movies recommended */]
#                 }
#                 """
#     llm_model_choice = "gpt-4o-mini"
#
#     generated_text = utils.generate_text(prompt,llm_model_choice,OPENAI_API_KEY)
#     movies_removed, movies_reco = utils.extract_movies(generated_text)
#     recall_model, precision_model = utils.calculate_recall_precision(ground_truth, topk_movies_rec)


    # if movies_removed is not None and movies_reco is not None:
    #     movie_map_rem = [movie_id for movie_id, title in ml_data["movieid_title"].items() if
    #                      movies_removed is not None and title in movies_removed]
    #
    #     movie_real_rem = [
    #         list(ml_data["movie_mapping"].values())[  # Retrieve the keys (movie IDs) of the movie_mapping dictionary
    #             list(ml_data["movie_mapping"].keys()).index(movie)
    #             # Get the index of the current movie ID in the values of movie_mapping
    #         ] for movie in movie_map_rem  # Iterate over each movie ID in topk_movies_rec
    #     ]
    #     rec_movie_ids_filtered = [movie_id for movie_id in topk_movies_rec if
    #                               movie_id not in movie_real_rem]  # removing movie ids not recommended
    #
    #     movie_names_reco = [title for title in movies_reco if
    #                         any(fuzz.partial_ratio(title, movie_title) >= 80 for movie_title in ml_data["movieid_title"].values())]
    #     print("movie names recommended by GPT", movie_names_reco)
    #     movie_map_reco = [movie_id for movie_id, title in ml_data["movieid_title"].items() if
    #                       movie_names_reco is not None and title in movie_names_reco]
    #
    #     movie_real_reco = [
    #         list(["movie_mapping"].values())[list(["movie_mapping"].keys()).index(movie)] for movie in movie_map_reco]
    #
    #     recall_model, precision_model = utils.calculate_recall_precision(ground_truth, topk_movies_rec)
    #     rec_movie_ids_total = list(set(topk_movies_rec + movie_real_reco))  # ensures only unique ids from both
    #     rec_movie_ids_final = list(set(rec_movie_ids_filtered + movie_real_reco))
    #     if len(rec_movie_ids_final) >= topK:
    #         rec_movie_ids_final = rec_movie_ids_final[:topK]
    #     else:
    #         remaining = topK - len(rec_movie_ids_final)
    #         rec_movie_ids_final = rec_movie_ids_final + movie_real_rem[:remaining]
    #         # Iterate through each id in movie_names_reco
    #     # for movie in movie_real_reco:
    #     #     # Check if the id exists in titles_liked_json
    #     #     if movie in hrated_movies:
    #     #         # Check if the id exists in rec_movie_ids_final
    #     #         if movie in rec_movie_ids_final:
    #     #             occurences += 1
    #     #             print("occurences", occurences)
    #     #             user_instance = 1
    #     # if user_instance == 1:
    #     #     user_count_instances += 1
    #     #     print("user instances", user_count_instances)
    #
    #     if len(rec_ids_real) > len(rec_movie_ids_total):
    #         cut_off_compare = len(rec_movie_ids_total)
    #     else:
    #         cut_off_compare = len(rec_ids_real)
    #
    #     recall_lightgcn, precision_lightgcn = utils.calculate_recall_precision(ground_truth, rec_ids_real[:cut_off_compare])


if __name__ == "__main__":
    uvicorn.run("app:app", port=8000, log_level="info", reload=True)
