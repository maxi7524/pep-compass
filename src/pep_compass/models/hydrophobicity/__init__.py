# Hydrophobicity Predictor Module

# This module provides hydrophobicity prediction capabilities using established amino acid hydrophobicity scales.

## Classes

# - `HydrophobicityPredictor`: Main predictor class for calculating peptide hydrophobicity scores

# ## Supported Scales

# - **Eisenberg scale**: Normalized consensus hydrophobicity scale from Eisenberg et al. (1984)

## Usage

# ```python
# from pep_compass.models.hydrophobicity.HydrophobicityPredictor import HydrophobicityPredictor

# # Initialize predictor
# predictor = HydrophobicityPredictor(scale="eisenberg")

# # Predict hydrophobicity for sequences
# sequences = ["KLLLKLLKKLLKLLK", "RRRRRRR"]
# scores = predictor.predict(sequences)
# ```