import json
import csv
from pathlib import Path
import pandas as pd
from .DatasetPromptBuilder import DatasetPromptBuilder

class MoviesBooksBuilder(DatasetPromptBuilder):
    def build(self):
        self.movies_prompt()
        #self.ratings_prompt()
        #self.users_prompt()

    def convert_csv_to_jsonl(self, csv_input_path, jsonl_output_path):
        with open(csv_input_path, 'r', encoding='utf-8') as csvfile, \
                open(jsonl_output_path, 'w', encoding='utf-8') as jsonlfile:
            reader = csv.DictReader(csvfile, delimiter=',')

            for row in reader:
                new_entry = {
                    "record": row["text"],
                    "partOfDataset": "Yes" if row["label"] == "1" else "No"
                }
                jsonlfile.write(json.dumps(new_entry, ensure_ascii=False) + '\n')

    def _create_templates(self, data, template_func):
        data["template"] = data["record"].apply(template_func)
        data["template_pos"] = data.apply(
            lambda row: f"{row['template']} {'Yes' if row['partOfDataset'] == 'Yes' else 'No'}", axis=1
        )
        data["template_neg"] = data.apply(
            lambda row: f"{row['template']} {'No' if row['partOfDataset'] == 'Yes' else 'Yes'}", axis=1
        )
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
        movies_dat = (current_dir / "../datasets/ml-books/movies_books.csv").resolve()
        movies_jsonl = movies_dat.with_suffix(".jsonl")

        def process_movie_line(line):
            parts = line.split("::")
            movie_id = int(parts[0])
            return {
                "movieId": movie_id,
                "record": line,
                "partOfDataset": "Yes"
            }

        self.convert_csv_to_jsonl(movies_dat, movies_jsonl)
        data = pd.read_json(movies_jsonl, orient="records", lines=True)
        data = self._create_templates(data, lambda record: f"Is {record} part of the dataset movielens?")
        #data = self._add_distraction_and_backslash(data)

        prompt_folder = Path(f"./prompt_datasets/{self.dataset_path}")
        prompt_folder.mkdir(parents=True, exist_ok=True)
        data.to_json(f"{prompt_folder}/movies_books_prompts.jsonl", orient="records", lines=True, force_ascii=False)

