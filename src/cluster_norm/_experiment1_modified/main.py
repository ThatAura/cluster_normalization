#!/usr/bin/env python3

import subprocess
import sys
import argparse

def run_script(script_name, args_list):
    """Esegue uno script come un sottoprocesso e gestisce l'output."""
    print(f"\n{'='*30}")
    print(f"Esecuzione di: {script_name} {' '.join(map(str, args_list))}")
    print(f"{'='*30}")
    try:
        # Costruisci il comando completo
        command = [sys.executable, script_name] + args_list
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        if stdout:
            print("Output:")
            print(stdout)
        if stderr:
            print("Errori:")
            print(stderr)

        if process.returncode == 0:
            print(f"\nScript '{script_name}' completato con successo.")
            return True
        else:
            print(f"\nErrore durante l'esecuzione di '{script_name}'. Codice di uscita: {process.returncode}")
            return False

    except FileNotFoundError:
        print(f"\nErrore: Lo script '{script_name}' non è stato trovato.")
        return False
    except Exception as e:
        print(f"\nSi è verificato un errore durante l'esecuzione di '{script_name}': {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esegue in sequenza create_prompt_dataset, harvest e experiment.")

    # Argomenti comuni a tutti o ad alcuni script
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Modello da usare")
    parser.add_argument("--dataset", type=str, default="ml", help="Nome del dataset (ml, imdb...)")
    parser.add_argument("--num_random_words", type=int, default=5, help="Numero di parole casuali")
    parser.add_argument("--layers", nargs='+', type=int, help="Lista di layer per harvest", default=None)

    # Argomenti specifici per create_prompt_dataset
    parser.add_argument("--k", type=int, default=5, help="Numero di gruppi per create_prompt_dataset")

    # Altri argomenti specifici per harvest o experiment se necessario
    # parser.add_argument("--altro_argomento_harvest", type=str, help="...")
    # parser.add_argument("--altro_argomento_experiment", type=float, help="...")

    args = parser.parse_args()

    create_prompt_script = "create_prompt_datasets.py"
    harvest_script = "harvest.py"
    experiment_script = "experiment.py"

    # Prepara le liste di argomenti per ciascun script
    create_prompt_args = ["--dataset", args.dataset, "--k", str(args.k)]
    harvest_args = ["--model", args.model, "--dataset", args.dataset]
    if args.layers:
        harvest_args.extend(["--layers", *map(str, args.layers)])
    experiment_args = ["--model", args.model, "--dataset", args.dataset, "--num_random_words", str(args.num_random_words)]

    # Esegui create_prompt_dataset
    if run_script(create_prompt_script, create_prompt_args):
        # Esegui harvest
        if run_script(harvest_script, harvest_args):
            # Esegui experiment
            run_script(experiment_script, experiment_args)
        else:
            print(f"\nL'esecuzione di '{harvest_script}' è fallita.")
    else:
        print(f"\nL'esecuzione di '{create_prompt_script}' è fallita.")

    print("\nProcesso sequenziale completato.")