import numpy as np
from typing import List, Union
from seqme.models import Hydrophobicity


class HydrophobicityPredictor:
    """
    Hydrophobicity predictor using the Eisenberg scale.
    Compatible interface with other predictors (APEX, BattleAMP, etc.).
    """
    
    def __init__(self, scale: str = "eisenberg"):
        """
        Initialize the Hydrophobicity predictor using seqme.
        
        Args:
            scale (str): Hydrophobicity scale to use. Supports scales available in seqme.
        """
        self.scale = scale
        
        # Initialize seqme Hydrophobicity model
        try:
            self.hydrophobicity_model = Hydrophobicity(scale=scale)
        except Exception as e:
            # Fallback to eisenberg if the scale is not available
            print(f"Warning: Scale '{scale}' not available, falling back to 'eisenberg'. Error: {e}")
            self.hydrophobicity_model = Hydrophobicity(scale="eisenberg")
            self.scale = "eisenberg"
        
        # Set pathogen list to match other predictors
        self.pathogen_list = ["Hydrophobicity"]

    def predict(self, seq_list: List[str], use_tqdm: bool = False) -> np.ndarray:
        """
        Predict hydrophobicity scores for a list of peptide sequences using seqme.
        
        Args:
            seq_list (list): List of peptide sequences as strings.
            use_tqdm (bool): Whether to show progress bars (not used in this implementation).
            
        Returns:
            np.ndarray: Predicted hydrophobicity scores with shape (N, 1) where N is the number of sequences.
                       Higher values indicate more hydrophobic peptides.
        """
        scores = []
        
        for seq in seq_list:
            try:
                # Use seqme's Hydrophobicity model to score the sequence
                score = self.hydrophobicity_model.__call__(seq)
                scores.append(score)
            except Exception as e:
                print(f"Warning: Error scoring sequence '{seq}': {e}. Using 0.0")
                scores.append(0.0)
        
        # Return as numpy array with shape (N, 1) to match other predictors
        return np.array(scores).reshape(-1, 1)

    def get_available_scales(self) -> List[str]:
        """
        Get list of available hydrophobicity scales from seqme.
        
        Returns:
            List[str]: Available scales in seqme.
        """
        try:
            return self.hydrophobicity_model.available_scales
        except:
            return ["eisenberg"]  # fallback
    
    def get_scale_info(self) -> dict:
        """
        Get information about the hydrophobicity scale being used.
        
        Returns:
            dict: Information about the scale including name and available scales.
        """
        return {
            "current_scale": self.scale,
            "available_scales": self.get_available_scales(),
            "description": f"Hydrophobicity predictor using seqme with {self.scale} scale",
            "seqme_model": str(type(self.hydrophobicity_model).__name__)
        }


if __name__ == "__main__":
    # Test the hydrophobicity predictor with seqme
    print("=== seqme Hydrophobicity Predictor Test ===")
    
    try:
        predictor = HydrophobicityPredictor(scale="eisenberg")
        
        # Test sequences
        test_sequences = [
            "KLLLKLLKKLLKLLK",  # More hydrophobic (lots of L)
            "FLPIIAKLLGLL",     # Mixed hydrophobicity
            "WLGHFTVRK",        # Mixed
            "RRRRRRR",          # Very hydrophilic (all R)
            "LLLLLLL",          # Very hydrophobic (all L)
            "GGGGGGG"           # Neutral (all G)
        ]
        
        print(f"Scale info: {predictor.get_scale_info()}")
        print(f"Pathogen list: {predictor.pathogen_list}")
        
        print("\nTesting predictions:")
        predictions = predictor.predict(test_sequences)
        
        for i, (seq, score) in enumerate(zip(test_sequences, predictions.flatten())):
            print(f"{i+1:2d}. {seq:15s} -> Hydrophobicity: {score:6.3f}")
        
        print(f"\nPredictions shape: {predictions.shape}")
        print("✓ seqme Hydrophobicity predictor test completed successfully!")
        
    except Exception as e:
        print(f"Error testing hydrophobicity predictor: {e}")
        import traceback
        traceback.print_exc()