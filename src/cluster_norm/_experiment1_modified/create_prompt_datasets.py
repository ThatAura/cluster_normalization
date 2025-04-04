import argparse
from prompt_datasets_generators.MovieLensGenerator import MovieLensBuilder
# eventualmente aggiungi altri builder qui

def get_builder(dataset_name, dataset_path, k):
    if dataset_name == "ml":
        return MovieLensBuilder(dataset_path, k)
    # elif dataset_name == "imdb":
    #     return ImdbBuilder(dataset_path, k)
    else:
        raise ValueError(f"Dataset {dataset_name} non supportato.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml", help="Nome del dataset (ml, imdb...)")
    parser.add_argument("--k", type=int, default=5, help="Numero di gruppi")
    args = parser.parse_args()

    builder = get_builder(args.dataset, args.dataset, args.k)
    builder.build()
