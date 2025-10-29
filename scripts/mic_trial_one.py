#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import pandas as pd
import argparse

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.pep_compass.models.apex.APEX_predictor import PredictorAPEX

def predict_mic_for_csv(file_path, predictor):
    """
    Loads a CSV file with a 'mutant' column, predicts MICs for all mutants,
    and saves the results to a new CSV file (_with_predictions.csv) with columns
    predicted_mic_<bacteria> for each bacterium in predictor.pathogen_list.
    """

    df = pd.read_csv(file_path)
    all_sequences = df['mutant'].tolist()
    print(f"Predicting MICs for {len(all_sequences)} mutants in {file_path}...")

    with torch.no_grad():
        predictions = predictor.predict(all_sequences)  # numpy array (num_mutants, num_pathogens)

    for j, pathogen in enumerate(predictor.pathogen_list):
        col_name = f'predicted_mic_{pathogen}'
        if col_name not in df.columns:
            df[col_name] = None
        df[col_name] = predictions[:, j]

    output_file = file_path.replace(".csv", "_with_predictions.csv")
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}!")

    return df

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Predict MICs for mutant peptides CSV using APEX models.")
    parser.add_argument("--file", type=str, required=True, help="Path to the CSV file with mutant sequences")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for PredictorAPEX")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    predictor = PredictorAPEX(device, batch_size=args.batch_size, path="all")

    predict_mic_for_csv(args.file, predictor)

    # Example usage:
    # python predict_mic.py --file /path/to/mutants.csv --batch_size 64
