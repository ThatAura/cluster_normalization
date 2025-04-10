import argparse
from prompt_datasets_generators.MovieLensGenerator import MovieLensBuilder
from prompt_datasets_generators.MoviesBooksGenerator import MoviesBooksBuilder
# eventualmente aggiungi altri builder qui

def get_builder(dataset_name, k):
    if dataset_name == "ml":
        return MovieLensBuilder(dataset_name, k)
    elif dataset_name == "ml-bk":
        return MoviesBooksBuilder(dataset_name, k)
    else:
        raise ValueError(f"Dataset {dataset_name} non supportato.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml-bk", help="Nome del dataset (ml, imdb...)")
    parser.add_argument("--k", type=int, default=5, help="Numero di gruppi")
    args = parser.parse_args()

    builder = get_builder(args.dataset, args.k)
    builder.build()
