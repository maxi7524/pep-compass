import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import math
from tqdm import tqdm
import sys

# Add local tools to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'tools'))

from tools.MultiBranchCNN import CNNimportFtsDataSetsPoNe, CNNstandardInputOutput
from config import def_fts, def_bias, def_scale


class PredictorMBCAttention:
    """
    MBC Attention predictor following the same interface as APEX and BattleAMP predictors.
    Predicts antimicrobial activity (MIC values) for peptide sequences using Multi-Branch CNN.
    """

    def __init__(self, device="cpu", batch_size=3000, path="default"):
        """
        Initialize the MBC Attention predictor.
        
        Args:
            device (str): Device to use for computation ("cpu" or "cuda"). 
                         Note: TensorFlow will handle GPU allocation automatically.
            batch_size (int): Batch size for processing sequences.
            path (str): Path configuration - currently only "default" supported.
        """
        self.device = device
        self.batch_size = batch_size
        self.path = path
        
        if path != "default":
            raise ValueError("Path option not recognized. Currently only 'default' is supported.")
            
        # Load the pre-trained MBC Attention model
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(self.file_dir, "model")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MBC Attention model not found at {model_path}")
            
        # Configure TensorFlow to use GPU if available and requested
        if device == "cuda":
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    # Enable memory growth to avoid allocating all GPU memory at once
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(f"GPU configuration error: {e}")
            else:
                print("CUDA requested but no GPUs available, using CPU")
        
        self.model = keras.models.load_model(model_path)
        
        # Set pathogen list to match APEX structure (single model predicting general antimicrobial activity)
        self.pathogen_list = ["E. coli"]
        
        # Store feature configuration
        self.ft_list = def_fts
        self.bias = def_bias
        self.scale = def_scale

    def predict(self, seq_list, use_tqdm: bool = False):
        """
        Predict antimicrobial activity (MIC values) for a list of peptide sequences.
        
        Args:
            seq_list (list): List of peptide sequences as strings.
            use_tqdm (bool): Whether to show progress bars.
            
        Returns:
            np.ndarray: Predicted MIC values with shape (N, 1) where N is the number of sequences.
                       Values are in μM units (normal MIC, not log10).
        """
        data_len = len(seq_list)
        
        # Progress bar setup
        pbar = tqdm(total=data_len, desc="Processing sequences") if use_tqdm else None
        
        # Process sequences in batches
        batch_iter = range(int(math.ceil(data_len / float(self.batch_size))))
        all_predictions = []
        
        for i in batch_iter:
            # Get batch of sequences
            seq_batch = seq_list[i * self.batch_size : (i + 1) * self.batch_size]
            
            # Create fastas DataFrame directly from peptide list (no temp files)
            peptide_ids = [f"peptide_{j}" for j in range(len(seq_batch))]
            fastas = pd.DataFrame({
                "ID": peptide_ids, 
                "SEQUENCE": seq_batch
            })
            
            # Generate CNN input datasets directly from DataFrame
            sets = CNNimportFtsDataSetsPoNe(fastas, ft_list=self.ft_list, target=None)
            
            # Prepare input for CNN
            X, Y = CNNstandardInputOutput(sets)
            
            # Make predictions
            batch_predictions = self.model.predict(X)
            
            # Convert predictions from scaled values back to log10 MIC values
            pred_log10_mic = batch_predictions / self.scale - self.bias
            
            # Convert from log10 MIC to normal MIC values (10^(-log10_mic))
            # Note: The model outputs -log10(MIC), so we need to negate and then take 10^x
            pred_mic = 10 ** (-pred_log10_mic)
            
            all_predictions.append(pred_mic)
            
            # Update progress bar
            if pbar is not None:
                pbar.update(len(seq_batch))
        
        if pbar is not None:
            pbar.close()
            
        # Concatenate all batch predictions
        predictions = np.vstack(all_predictions)
        
        return predictions