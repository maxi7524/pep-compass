#!/usr/bin/env python3
"""
BattleAMP Single Peptide Prediction Test
Tests a single target peptide sequence with BattleAMP
"""

import sys
import os
import warnings
import time
import numpy as np

print("BattleAMP Single Peptide Test")
print("=" * 40)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['RDKIT_QUIET'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add the src directory to the Python path
sys.path.insert(0, '/home/kjurasz/pep-compass/src')

print("Loading BattleAMP model...")
start_time = time.time()

try:
    from pep_compass.optimization.black_box.battleamp_black_box import BattleAMPBlackBox
    load_time = time.time() - start_time
    print(f"✓ BattleAMP imported successfully ({load_time:.1f}s)")
except ImportError as e:
    print(f"✗ Failed to import BattleAMP: {e}")
    sys.exit(1)

def test_single_peptide():
    """Test BattleAMP prediction on a single peptide"""
    
    # Determine device
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    
    print(f"Device: {device}")
    
    # Initialize BattleAMP
    try:
        print("Initializing BattleAMP...")
        init_start = time.time()
        black_box = BattleAMPBlackBox(device=device)
        init_time = time.time() - init_start
        print(f"✓ BattleAMP initialized successfully ({init_time:.1f}s)")
    except Exception as e:
        print(f"✗ Failed to initialize BattleAMP: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Test peptides - you can change this list
    test_peptides = [
        'SGWGCCCCCCCWCY'
    ]
    
    print(f"\nTesting {len(test_peptides)} peptides:")
    
    results = []
    
    for i, peptide in enumerate(test_peptides, 1):
        print(f"\n{i}. Testing: {peptide}")
        print(f"   Length: {len(peptide)} amino acids")
        
        try:
            # Prepare input as numpy array
            pred_start = time.time()
            peptide_array = np.array([peptide])
            
            # Get prediction
            predictions = black_box(peptide_array)
            pred_time = time.time() - pred_start
            
            # Extract score - handle different return formats
            if hasattr(predictions, 'flatten'):
                raw_score = predictions.flatten()[0]
            elif hasattr(predictions, '__getitem__'):
                raw_score = predictions[0]
                if isinstance(raw_score, (list, tuple)):
                    raw_score = raw_score[0]
            else:
                raw_score = float(predictions)
            
            # Apply sign reversal for optimization framework consistency
            reversed_score = -float(raw_score)
            
            print(f"   ✓ Prediction completed ({pred_time:.3f}s)")
            print(f"   Raw BattleAMP output: {raw_score:.6f}")
            print(f"   Sign-reversed score: {reversed_score:.6f}")
            
            # Interpretation
            if reversed_score > 0:
                interpretation = "HIGHER antimicrobial activity"
            else:
                interpretation = "LOWER antimicrobial activity"
            print(f"   → Prediction: {interpretation}")
            
            results.append({
                'peptide': peptide,
                'raw_score': raw_score,
                'our_score': reversed_score,
                'interpretation': interpretation
            })
            
        except Exception as e:
            print(f"   ✗ Prediction failed: {e}")
            results.append({
                'peptide': peptide,
                'raw_score': None,
                'our_score': None,
                'interpretation': 'FAILED'
            })
    
    # Summary
    print(f"\n" + "=" * 40)
    print("SUMMARY RESULTS:")
    print("=" * 40)
    
    for result in results:
        status = "✓" if result['our_score'] is not None else "✗"
        peptide = result['peptide']
        score = f"{result['our_score']:.6f}" if result['our_score'] is not None else "FAILED"
        interpretation = result['interpretation']
        
        print(f"{status} {peptide}")
        print(f"    Score: {score}")
        print(f"    Prediction: {interpretation}")
        print()
    
    # Find best peptide
    valid_results = [r for r in results if r['our_score'] is not None]
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['our_score'])
        print(f"Best peptide by BattleAMP score:")
        print(f"  {best_result['peptide']} (Score: {best_result['our_score']:.6f})")
    
    return results

def analyze_peptide_properties(sequence):
    """Analyze amino acid composition and properties"""
    from collections import Counter
    
    print(f"\n--- Sequence Analysis for {sequence} ---")
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
    results = test_single_peptide()
    
    # Optionally analyze the first peptide in detail
    if results and results[0]['peptide']:
        analyze_peptide_properties(results[0]['peptide'])
    
    print(f"\nTest completed!")