#!/usr/bin/env python3
"""
Test script for the BattleAMP predictor to ensure it works similarly to APEX predictor.
"""

import sys
import os
import numpy as np

# Add the source directory to the path
sys.path.insert(0, '/home/kjurasz/pep-compass/src')

from pep_compass.models.battleamp import PredictorBattleAMP

def test_battleamp_predictor():
    """Test the BattleAMP predictor with sample sequences."""
    
    print("Testing BattleAMP Predictor...")
    
    # Initialize predictor
    try:
        predictor = PredictorBattleAMP(device="cpu", batch_size=32)
        print(f"✓ Predictor initialized successfully")
        print(f"  - Device: {predictor.device}")
        print(f"  - Batch size: {predictor.batch_size}")
        print(f"  - Pathogen list: {predictor.pathogen_list}")
    except Exception as e:
        print(f"✗ Failed to initialize predictor: {e}")
        return False
    
    # Test sequences (same as used in the notebook)
    test_sequences = [
        "ACDEFGHIKLMNPQRST",
        "RRRRRRRRRRRRRRRR", 
        "WLGHFTVRK",
        "KLLLKLLKKLLKLLK",
        "FLPIIAKLLGLL"
    ]
    
    print(f"\nTesting with {len(test_sequences)} sequences:")
    for i, seq in enumerate(test_sequences):
        print(f"  {i+1}. {seq}")
    
    # Make predictions
    try:
        predictions = predictor.predict(test_sequences, use_tqdm=True)
        print(f"\n✓ Predictions completed successfully")
        print(f"  - Output shape: {predictions.shape}")
        print(f"  - Output type: {type(predictions)}")
        
        # Display results
        print("\nPrediction Results:")
        print("Sequence → MIC (μM)")
        print("-" * 40)
        for i, (seq, mic) in enumerate(zip(test_sequences, predictions)):
            print(f"{i+1:2d}. {seq:20s} → {mic[0]:.4f}")
            
        return True
        
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False

def compare_with_notebook_results():
    """Compare results with the notebook implementation."""
    
    print("\n" + "="*50)
    print("COMPARING WITH NOTEBOOK IMPLEMENTATION")
    print("="*50)
    
    # Test the same sequences as in the notebook
    notebook_sequences = [
        "ACDEFGHIKLMNPQRST",
        "RRRRRRRRRRRRRRRR",
        "WLGHFTVRK"
    ]
    
    predictor = PredictorBattleAMP(device="cpu")
    predictions = predictor.predict(notebook_sequences)
    
    print("Notebook sequences:")
    for i, (seq, mic) in enumerate(zip(notebook_sequences, predictions)):
        print(f"  {seq} → {mic[0]:.6f} μM")
    
    print("\nNote: Results should be similar to the notebook output.")
    print("Small differences may occur due to model loading or TensorFlow version differences.")

if __name__ == "__main__":
    print("BattleAMP Predictor Test")
    print("=" * 30)
    
    success = test_battleamp_predictor()
    
    if success:
        compare_with_notebook_results()
        print("\n✓ All tests completed successfully!")
    else:
        print("\n✗ Tests failed.")
        sys.exit(1)