#!/usr/bin/env python3
"""
BattleAMP Peptide Prediction Test Script
Tests the target peptide sequence 'TGDNDDDDDDDDDD'
"""

import sys
import os
import warnings
import time
import pandas as pd
import numpy as np

print("BattleAMP Batch Prediction for GFN AL CS")
print("=" * 40)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['RDKIT_QUIET'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add the src directory to the Python path
sys.path.insert(0, '/home/kjurasz/pep-compass/src')

print("Loading BattleAMP model (this may take a moment)...")
start_time = time.time()

try:
    from pep_compass.optimization.black_box.battleamp_black_box import BattleAMPBlackBox
    load_time = time.time() - start_time
    print(f"✓ BattleAMP imported successfully ({load_time:.1f}s)")
except ImportError as e:
    print(f"✗ Failed to import BattleAMP: {e}")
    sys.exit(1)

def process_gfn_csv():
    """Process GFN AL CS CSV file and add BattleAMP predictions"""
    
    # File paths
    input_file = '/home/kjurasz/pep-compass/results/gfn_al_cs_FL14/gfn_al_cs_FL14_seed_1.csv'
    output_file = input_file.replace('.csv', '_with_battleamp.csv')
    
    # Load CSV
    print(f"Loading CSV file: {input_file}")
    try:
        df = pd.read_csv(input_file)
        print(f"✓ Loaded {len(df)} sequences")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"✗ Failed to load CSV: {e}")
        return None
    
    # Initialize BattleAMP
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    
    print(f"Device: {device}")
    
    try:
        print("Initializing BattleAMP (this may take 30-60 seconds)...")
        init_start = time.time()
        black_box = BattleAMPBlackBox(device=device)
        init_time = time.time() - init_start
        print(f"✓ BattleAMP initialized successfully ({init_time:.1f}s)")
    except Exception as e:
        print(f"✗ Failed to initialize BattleAMP: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Process sequences in batches
    sequences = df['Sequence'].tolist()
    batch_size = 50  # Process in smaller batches to avoid memory issues
    all_predictions = []
    
    print(f"\nProcessing {len(sequences)} sequences in batches of {batch_size}...")
    
    for i in range(0, len(sequences), batch_size):
        batch_end = min(i + batch_size, len(sequences))
        batch_sequences = sequences[i:batch_end]
        
        print(f"Processing batch {i//batch_size + 1}/{(len(sequences) + batch_size - 1)//batch_size} "
              f"(sequences {i+1}-{batch_end})")
        
        try:
            # Prepare batch as numpy array
            batch_array = np.array(batch_sequences)
            
            # Get predictions
            batch_start = time.time()
            predictions = black_box(batch_array)
            batch_time = time.time() - batch_start
            
            # Convert predictions to flat list
            if hasattr(predictions, 'flatten'):
                batch_predictions = predictions.flatten().tolist()
            elif hasattr(predictions, 'tolist'):
                flat_preds = predictions.tolist()
                # Flatten if nested
                if isinstance(flat_preds[0], list):
                    batch_predictions = [item for sublist in flat_preds for item in sublist]
                else:
                    batch_predictions = flat_preds
            elif hasattr(predictions, '__iter__'):
                batch_predictions = list(predictions)
                # Flatten if nested
                if batch_predictions and isinstance(batch_predictions[0], (list, tuple)):
                    batch_predictions = [item for sublist in batch_predictions for item in sublist]
            else:
                batch_predictions = [predictions] * len(batch_sequences)
            
            # Apply sign reversal for optimization framework consistency
            batch_predictions_reversed = [-float(score) for score in batch_predictions]
            all_predictions.extend(batch_predictions_reversed)
            
            print(f"  ✓ Batch completed ({batch_time:.2f}s, {len(batch_sequences)/batch_time:.1f} seq/s)")
            
        except Exception as e:
            print(f"  ✗ Batch failed: {e}")
            # Fill with NaN for failed predictions
            all_predictions.extend([np.nan] * len(batch_sequences))
    
    # Add predictions to dataframe
    df['our_score'] = all_predictions
    
    # Save results
    try:
        df.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to: {output_file}")
        
        # Show statistics
        valid_predictions = df['our_score'].dropna()
        print(f"\nStatistics:")
        print(f"  Total sequences: {len(df)}")
        print(f"  Valid predictions: {len(valid_predictions)}")
        print(f"  Failed predictions: {len(df) - len(valid_predictions)}")
        
        if len(valid_predictions) > 0:
            print(f"  Score range: {valid_predictions.min():.6f} to {valid_predictions.max():.6f}")
            print(f"  Mean score: {valid_predictions.mean():.6f}")
            print(f"  Median score: {valid_predictions.median():.6f}")
            
            # Show top 5 sequences by BattleAMP score
            print(f"\nTop 5 sequences by BattleAMP score:")
            top_sequences = df.nlargest(5, 'our_score')
            for idx, row in top_sequences.iterrows():
                print(f"  {row['Sequence']} -> {row['our_score']:.6f}")
        
        return output_file
        
    except Exception as e:
        print(f"✗ Failed to save results: {e}")
        return None

def analyze_peptide_properties(sequence):
    """Analyze amino acid composition and properties"""
    from collections import Counter
    
    print(f"\n--- Sequence Analysis ---")
    aa_counts = Counter(sequence)
    
    print("Amino acid composition:")
    for aa, count in sorted(aa_counts.items()):
        percentage = (count / len(sequence)) * 100
        print(f"  {aa}: {count} ({percentage:.1f}%)")
    
    # Calculate properties
    positive_aa = "RHK"
    negative_aa = "DE"
    hydrophobic_aa = "AILMFPWYV"
    polar_aa = "NQST"
    
    pos_count = sum(1 for aa in sequence if aa in positive_aa)
    neg_count = sum(1 for aa in sequence if aa in negative_aa)
    hydrophobic_count = sum(1 for aa in sequence if aa in hydrophobic_aa)
    polar_count = sum(1 for aa in sequence if aa in polar_aa)
    
    net_charge = pos_count - neg_count
    
    print(f"\nProperties:")
    print(f"  Net charge: {net_charge:+d} (Pos: {pos_count}, Neg: {neg_count})")
    print(f"  Hydrophobic residues: {hydrophobic_count}")
    print(f"  Polar residues: {polar_count}")

if __name__ == "__main__":
    output_file = process_gfn_csv()
    if output_file:
        print(f"\n✓ Processing completed successfully!")
        print(f"Output file: {output_file}")
    else:
        print("\n✗ Processing failed!")
        sys.exit(1)