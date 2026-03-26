import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import math
from tqdm import tqdm
from .utils import chopping, padding, onehot_encoding, prepare_CNN


class PredictorBattleAMP:
    """
    BattleAMP predictor following the same interface as APEX predictor.
    Predicts antimicrobial activity (MIC values) for peptide sequences.
    """

    def __init__(self, device="cpu", batch_size=3000, path="default"):
        """
        Initialize the BattleAMP predictor.
        
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
            
        # Load the pre-trained BattleAMP model
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(self.file_dir, "model.h5")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"BattleAMP model not found at {model_path}")
            
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
        self.pathogen_list = ["Gram -"]

    def predict(self, seq_list, use_tqdm: bool = False):
        """
        Predict antimicrobial activity (MIC values) for a list of peptide sequences.
        
        Args:
            seq_list (list): List of peptide sequences as strings.
            use_tqdm (bool): Whether to show progress bars.
            
        Returns:
            np.ndarray: Predicted MIC values with shape (N, 1) where N is the number of sequences.
                       Values are in μM units.
        """
        data_len = len(seq_list)
        
        # Progress bar setup
        pbar = tqdm(total=data_len, desc="Processing sequences") if use_tqdm else None
        
        # Process sequences in batches
        batch_iter = range(int(math.ceil(data_len / float(self.batch_size))))
        all_predictions = []
        
        for i in batch_iter:
            
            seq_batch = seq_list[i * self.batch_size : (i + 1) * self.batch_size]
            batch_input = prepare_CNN(seq_batch) 
            
            
            batch_predictions = self.model(batch_input).numpy()  
            all_predictions.append(batch_predictions)
            
         
            if pbar is not None:
                pbar.update(len(seq_batch))
        
        if pbar is not None:
            pbar.close()
            
     
        predictions = np.vstack(all_predictions)
        
        # Convert from log10 MIC to normal MIC values
        predictions = 10 ** predictions
        
        return predictions

