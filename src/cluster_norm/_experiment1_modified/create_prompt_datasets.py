import argparse
import json
from pathlib import Path

import pandas as pd


def create_movies_jsonl(df, output_file):
    with open(output_file, "w", encoding="utf-8") as outfile:
        for _, row in df.iterrows():
            movie_data = {
                "movieId": int(row["MovieID"]),  # Converti MovieID in intero
                "record": row["Title"]  # Prendi solo il titolo come record
            }
            outfile.write(json.dumps(movie_data) + "\n")  # Scrivi in JSONL senza doppia serializzazione


# Set up argument parsing
parser = argparse.ArgumentParser(description="Divide data into groups and apply random words.")
parser.add_argument("--dataset", type=str, default="ml-1m", help="daataset to use (default: imdb)")
parser.add_argument("--k", type=int, default=5, help="Number of groups to divide data into (default: 2)")
args = parser.parse_args()

# Retrieve the value of `k` from the arguments
k = args.k

# Carica il dataset dei movies
movies = pd.read_csv(args.dataset + "/movies.dat", sep="::", header=None, names=['MovieID', 'Title', 'Genres'], engine='python', encoding='ISO-8859-1')

#create_movies_jsonl(movies, args.dataset + "/movies.jsonl")

data = pd.read_json(args.dataset + "/movies.jsonl", orient="records", lines=True)

data["template"] = data["record"].apply(lambda record: f"{record} part of the dataset movielens? answer with yes or no")
data["template_pos"] = data["template"].apply(lambda x: f"Is {x}")
data["template_neg"] = data["template"].apply(lambda x: f"Is not {x}")

data["template_pos_bs"] = ""
data["template_neg_bs"] = ""

# Load random words from a JSON file
with open("random_words.json", "r") as file:
    random_words = json.load(file)["words"]

# Ensure enough words are available for the desired number of groups
if len(random_words) < k:
    raise ValueError(f"Insufficient words in the file to divide into {k} groups.")

# Select the first `k` words for grouping
group_words = {i: random_words[i] for i in range(k)}

# Randomly divide indices into `k` groups
groups = data.index.to_series().sample(frac=1).groupby(lambda x: x % k).indices

# Apply the words based on the assigned group
for group_num, indices in groups.items():
    word = group_words[group_num]
    for col in ["template_pos", "template_neg"]:
        data.loc[indices, f"{col}_bs"] = data.loc[indices, col].apply(lambda x: f"{x}. {word}")
        data.loc[indices, "distraction"] = word

prompt_folder = Path(f"./prompt_datasets/{args.dataset}")
prompt_folder.mkdir(parents=True, exist_ok=True)
data.to_json(f"{prompt_folder}/prompts_{k}.jsonl", orient="records", lines=True)



