import torch
from APEX_predictor import PredictorAPEX  # zakładam, że twój kod jest zapisany jako predictor_apex.py

if __name__ == "__main__":
    # Ustawienia
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Twój testowy peptyd
    peptide = "CELHFRQLPAVKARLLKLRIKWNRK"
    seq_list = [peptide]

    # Inicjalizacja modelu w wersji 'all'
    print("Loading APEX models (version: all)...")
    predictor = PredictorAPEX(device=device, path="all")

    # Predykcja
    print(f"Predicting MIC values for peptide: {peptide}")
    pred = predictor.predict(seq_list)

    # Wynik
    print("\nPredicted MICs (uM) for each pathogen:")
    for name, value in zip(predictor.pathogen_list, pred[0]):
        print(f"{name:60s} -> {value:.4f}")
