import json
from pathlib import Path
import pandas as pd
from DatasetPromptBuilder import DatasetPromptBuilder

class MovieLensBuilder(DatasetPromptBuilder):
    def build(self):
        self.movies_prompt()
        self.ratings_prompt()
        self.users_prompt()

    def _process_dat_to_jsonl(self, dat_path, jsonl_path, process_line_func):
        if jsonl_path.exists():
            print(f"Il file {jsonl_path} esiste.")
        else:
            with open(dat_path, "r", encoding="ISO-8859-1") as infile, open(jsonl_path, "w", encoding="utf-8") as outfile:
                for line in infile:
                    line = line.strip()
                    if not line:
                        continue
                    data = process_line_func(line)
                    outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            print(f"Creato il file {jsonl_path}")

    def _create_templates(self, data, template_func):
        data["template"] = data["record"].apply(template_func)
        data["template_pos"] = data["template"].apply(lambda x: f"{x} Yes")
        data["template_neg"] = data["template"].apply(lambda x: f"{x} No")
        return data

    def _add_distraction_and_backslash(self, data):
        with open("random_words.json", "r") as file:
            random_words = json.load(file)["words"]
        if len(random_words) < self.k:
            raise ValueError(f"Insufficient words for {self.k} groups.")
        group_words = {i: random_words[i] for i in range(self.k)}
        groups = data.index.to_series().sample(frac=1).groupby(lambda x: x % self.k).indices
        for group_num, indices in groups.items():
            word = group_words[group_num]
            for col in ["template_pos", "template_neg"]:
                data.loc[indices, f"{col}_bs"] = data.loc[indices, col].apply(lambda x: f"{x}. {word}")
                data.loc[indices, "distraction"] = word
        return data

    def movies_prompt(self):
        current_dir = Path(__file__).parent
        movies_dat = (current_dir / "../datasets/ml/movies.dat").resolve()
        movies_jsonl = movies_dat.with_suffix(".jsonl")

        def process_movie_line(line):
            parts = line.split("::")
            movie_id = int(parts[0])
            return {
                "movieId": movie_id,
                "record": line,
                "partOfDataset": "Yes"
            }

        self._process_dat_to_jsonl(movies_dat, movies_jsonl, process_movie_line)
        data = pd.read_json(movies_jsonl, orient="records", lines=True)
        data = self._create_templates(data, lambda record: f"Is {record} part of the dataset movielens?")
        data = self._add_distraction_and_backslash(data)

        prompt_folder = Path(f"./prompt_datasets/{self.dataset_path}")
        prompt_folder.mkdir(parents=True, exist_ok=True)
        data.to_json(f"{prompt_folder}/moviesPrompts_{self.k}.jsonl", orient="records", lines=True, force_ascii=False)

    def ratings_prompt(self):
        current_dir = Path(__file__).parent
        ratings_dat = (current_dir / "../datasets/ml/ratings.dat").resolve()
        ratings_jsonl = ratings_dat.with_suffix(".jsonl")

        def process_rating_line(line):
            parts = line.split("::")
            user_id, movie_id, rating, timestamp = map(int, parts)
            return {
                "userId": user_id,
                "movieId": movie_id,
                "rating": rating,
                "timestamp": timestamp,
                "record": line,
                "partOfDataset": "Yes"
            }

        self._process_dat_to_jsonl(ratings_dat, ratings_jsonl, process_rating_line)
        data = pd.read_json(ratings_jsonl, orient="records", lines=True)
        data = self._create_templates(data, lambda record: f"Is {record} part of the dataset movielens?")
        data = self._add_distraction_and_backslash(data)

        prompt_folder = Path(f"./prompt_datasets/{self.dataset_path}")
        prompt_folder.mkdir(parents=True, exist_ok=True)
        data.to_json(f"{prompt_folder}/ratingsPrompts_{self.k}.jsonl", orient="records", lines=True, force_ascii=False)

    def users_prompt(self):
        current_dir = Path(__file__).parent
        users_dat = (current_dir / "../datasets/ml/users.dat").resolve()
        users_jsonl = users_dat.with_suffix(".jsonl")

        def process_user_line(line):
            parts = line.split("::")
            user_id, gender, age, occupation, zip_code = parts
            return {
                "userId": int(user_id),
                "gender": gender,
                "age": int(age),
                "occupation": int(occupation),
                "zipCode": zip_code,
                "record": line,
                "partOfDataset": "Yes"
            }

        self._process_dat_to_jsonl(users_dat, users_jsonl, process_user_line)
        data = pd.read_json(users_jsonl, orient="records", lines=True)
        data = self._create_templates(data, lambda record: f"Is {record} part of the dataset movielens?")
        data = self._add_distraction_and_backslash(data)

        prompt_folder = Path(f"./prompt_datasets/{self.dataset_path}")
        prompt_folder.mkdir(parents=True, exist_ok=True)
        data.to_json(f"{prompt_folder}/usersPrompts_{self.k}.jsonl", orient="records", lines=True, force_ascii=False)