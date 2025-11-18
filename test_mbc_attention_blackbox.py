#!/usr/bin/env python3
"""
Test script for the MBC Attention black box to ensure it works correctly.
"""

import sys
import os
import numpy as np

# Add the source directory to the path
sys.path.insert(0, '/home/kjurasz/pep-compass/src')

from pep_compass.optimization.black_box.mbc_attention_black_box import MBCAttentionBlackBox

def test_mbc_attention_blackbox():
    """Test the MBC Attention black box with sample sequences."""
    
    print("Testing MBC Attention Black Box...")
    
    # Initialize black box
    try:
        black_box = MBCAttentionBlackBox(device="cpu", batch_size=32)
        print(f"✓ Black box initialized successfully")
        print(f"  - Device: {black_box.device}")
        print(f"  - Batch size: {black_box.batch_size}")
    except Exception as e:
        print(f"✗ Failed to initialize black box: {e}")
        return False
    
    # Test black box info
    try:
        info = black_box.get_black_box_info()
        print(f"\n✓ Black box info retrieved successfully")
        print(f"  - Name: {info.name}")
        print(f"  - Max sequence length: {info.max_sequence_length}")
        print(f"  - Alphabet: {info.alphabet}")
        print(f"  - Discrete: {info.discrete}")
    except Exception as e:
        print(f"✗ Failed to get black box info: {e}")
        return False
    
    # Test sequences as strings first
    test_sequence_strings = [
        "ACDEFGHIKLMNPQRST",
        "RRRRRRRRRRRRRRRR",
        "WLGHFTVRK",
        "KLLLKLLKKLLKLLK",
        "FLPIIAKLLGLL"
    ]
    
    print(f"\nTesting with {len(test_sequence_strings)} sequences:")
    for i, seq in enumerate(test_sequence_strings):
        print(f"  {i+1}. {seq}")
    
    # Convert to character arrays as expected by the black box
    test_sequences = [list(seq) for seq in test_sequence_strings]
    
    # Create padded array - all sequences need same length
    max_len = max(len(seq) for seq in test_sequences)
    padded_sequences = []
    for seq in test_sequences:
        # Pad with the last character or 'A' to avoid unknown characters
        padded = seq + ['A'] * (max_len - len(seq))
        padded_sequences.append(padded)
    
    x = np.array(padded_sequences)
    print(f"\nInput array shape: {x.shape}")
    
    # Make predictions
    try:
        predictions = black_box(x)
        print(f"\n✓ Predictions completed successfully")
        print(f"  - Output shape: {predictions.shape}")
        print(f"  - Output type: {type(predictions)}")
        
        # Display results
        print("\nBlack Box Prediction Results:")
        print("Sequence → Score")
        print("-" * 40)
        for i, (seq_str, score) in enumerate(zip(test_sequence_strings, predictions)):
            print(f"{i+1:2d}. {seq_str:20s} → {score[0]:.6f}")
        
        # Test cache
        print(f"\n✓ Cache contains {len(black_box.cache)} entries")
        if black_box.cache:
            print("First cache entry:")
            seq, score = black_box.cache[0]
            print(f"  Sequence: {seq}")
            print(f"  Score: {score}")
        
        # Clear cache
        black_box.clear_cache()
        print(f"✓ Cache cleared: {len(black_box.cache)} entries")
            
        return True
        
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_peptide_scorer():
    """Test the peptide scorer function directly."""
    
    print("\n" + "="*50)
    print("TESTING PEPTIDE SCORER DIRECTLY")
    print("="*50)
    
    black_box = MBCAttentionBlackBox(device="cpu")
    
    test_sequences = [
        "ACDEFGHIKLMNPQRST",
        "RRRRRRRRRRRRRRRR",
        "WLGHFTVRK"
    ]
    
    try:
        scores = black_box.peptide_scorer(test_sequences)
        print("Peptide scorer results:")
        for seq, score in zip(test_sequences, scores):
            print(f"  {seq} → {score:.6f}")
        return True
    except Exception as e:
        print(f"✗ Peptide scorer failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_other_blackboxes():
    """Compare with BattleAMP to show similar behavior."""
    
    print("\n" + "="*50)
    print("COMPARISON WITH OTHER BLACK BOXES")
    print("="*50)
    
    print("MBC Attention Black Box characteristics:")
    print("- Predicts E. coli antimicrobial activity")
    print("- Uses Multi-Branch CNN with 14 molecular descriptors")
    print("- Returns log2-transformed scores (negative = better)")
    print("- Single prediction value (no pathogen aggregation)")
    print("- Same interface as APEX/BattleAMP black boxes")

if __name__ == "__main__":
    print("MBC Attention Black Box Test")
    print("=" * 30)
    
    success1 = test_mbc_attention_blackbox()
    
    if success1:
        success2 = test_peptide_scorer()
        compare_with_other_blackboxes()
        
        if success1 and success2:
            print("\n✓ All tests completed successfully!")
        else:
            print("\n✗ Some tests failed.")
            sys.exit(1)
    else:
        print("\n✗ Tests failed.")
        sys.exit(1)