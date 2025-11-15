#!/usr/bin/env python3
"""
Example demonstrating EIPredBlackBox usage for peptide optimization.

This shows how to use EIPredBlackBox similarly to HydrAMPAPEXBlackBox,
including the full workflow from sequence encoding to optimization scoring.
"""
import os
import sys
import numpy as np

# Setup path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(script_dir, '..', '..', '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

print("=== EIPredBlackBox Example Usage ===\n")

def demo_eipred_blackbox():
    """Demonstrate EIPredBlackBox usage with example sequences."""
    
    from pep_compass.optimization.black_box.eipred_black_box import EIPredBlackBox
    
    # Create EIPred black box (similar to HydrAMPAPEXBlackBox)
    print("1. Creating EIPredBlackBox...")
    eipred_bb = EIPredBlackBox(
        mic_aggregate="mean",  # or "max" 
        batch_size=1000,
        evaluation_budget=float("inf")
    )
    print(f"   ✓ Black box created: {type(eipred_bb).__name__}")
    
    # Example peptide sequences to evaluate
    print("\n2. Testing with example sequences...")
    test_sequences = [
        "ACDEFGHIKLMNPQRSTVWY",  # All 20 amino acids
        "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",  # Antimicrobial peptide
        "MKTIIALSYIFCLVFA",  # Shorter peptide
        "GGG",  # Very short
        "RRWWRF",  # Cationic peptide
        "AAAEQLKTTRNAYHQKY" #should have -1.502
    ]
    
    print("   Test sequences:")
    for i, seq in enumerate(test_sequences, 1):
        print(f"     {i}. {seq} (length: {len(seq)})")
    
    # Get black box scores using peptide_scorer
    print("\n3. Computing black box scores...")
    scores = eipred_bb.peptide_scorer(test_sequences)
    
    print("   Results:")
    print(f"     Scores: {scores}")
    print(f"     Score range: {scores.min():.3f} to {scores.max():.3f}")
    print(f"     Mean score: {scores.mean():.3f}")
    
    # Interpret scores (higher = better antimicrobial activity)
    print("\n4. Sequence ranking by predicted activity:")
    ranked_indices = np.argsort(-scores)  # Sort descending (higher scores better)
    for rank, idx in enumerate(ranked_indices, 1):
        seq = test_sequences[idx]
        score = scores[idx]
        print(f"     {rank}. {seq[:30]}{'...' if len(seq) > 30 else ''} (score: {score:.3f})")
    
    return eipred_bb, test_sequences, scores

def demo_black_box_interface():
    """Demonstrate the full black box interface (like optimization would use)."""
    
    print("\n=== Black Box Interface Demo ===\n")
    
    from pep_compass.optimization.black_box.eipred_black_box import EIPredBlackBox
    
    # Create black box
    bb = EIPredBlackBox(mic_aggregate="mean")
    
    # In real optimization, sequences would be encoded as arrays
    # Here we'll simulate the _black_box method interface
    print("1. Black box info...")
    try:
        info = bb.get_black_box_info()
        print(f"   Black box info: {info}")
    except Exception as e:
        print(f"   Info not available: {e}")
    
    # Demonstrate context and cache behavior
    print("\n2. Testing context and cache...")
    test_seqs = ["KWKLFK", "RRWWRF", "ACDEF"]
    
    # Simulate what happens in optimization
    context = {}
    # scores_1 = bb.peptide_scorer(test_seqs)
    
    # print(f"   First call scores: {scores_1}")
    # print(f"   Cache size after first call: {len(bb.cache)}")
    
    # # Second call with same sequences (should use cache or be consistent)
    # scores_2 = bb.peptide_scorer(test_seqs)
    # print(f"   Second call scores: {scores_2}")
    # print(f"   Consistent results? {np.allclose(scores_1, scores_2)}")
    
    return bb

def compare_with_apex():
    """Compare EIPred and APEX black boxes side by side."""
    
    print("\n=== Comparison with APEX BlackBox ===\n")
    
    from pep_compass.optimization.black_box.eipred_black_box import EIPredBlackBox
    from pep_compass.optimization.black_box.apex_black_box import HydrAMPAPEXBlackBox
    
    # Test sequences
    seqs = ["KWKLFKKIEKVGQ", "RRWWRF", "ACDEFGHIKL"]
    
    print("1. Creating both black boxes...")
    eipred_bb = EIPredBlackBox(mic_aggregate="mean")
    apex_bb = HydrAMPAPEXBlackBox(
        mic_aggregate="mean", 
        device="cpu",
        jacobian_eps=1e-6, 
        field_eps=1e-6
    )
    
    print("2. Getting predictions from both models...")
    eipred_scores = eipred_bb.peptide_scorer(seqs)
    apex_scores = apex_bb.peptide_scorer(seqs)
    
    print("\n3. Results comparison:")
    print("   Sequence                    EIPred Score    APEX Score")
    print("   " + "="*55)
    for i, seq in enumerate(seqs):
        seq_display = seq[:20] + "..." if len(seq) > 20 else seq
        print(f"   {seq_display:<25} {eipred_scores[i]:8.3f}      {apex_scores[i]:8.3f}")
    
    print(f"\n   EIPred range: {eipred_scores.min():.3f} to {eipred_scores.max():.3f}")
    print(f"   APEX range:   {apex_scores.min():.3f} to {apex_scores.max():.3f}")
    
    # Check if rankings are similar
    eipred_ranking = np.argsort(-eipred_scores)
    apex_ranking = np.argsort(-apex_scores)
    
    print(f"\n   EIPred ranking: {eipred_ranking + 1}")  # 1-indexed for display
    print(f"   APEX ranking:   {apex_ranking + 1}")
    print(f"   Rankings match? {np.array_equal(eipred_ranking, apex_ranking)}")

if __name__ == "__main__":
    try:
        # Run the main demo
        bb, seqs, scores = demo_eipred_blackbox()
        
        # Show interface demo
        demo_black_box_interface()
        
        # Compare with APEX
        compare_with_apex()
        
        print("\n🎉 EIPredBlackBox example completed successfully!")
        print("\nNext steps:")
        print("- Use EIPredBlackBox in optimization algorithms")
        print("- Compare optimization results with HydrAMPAPEXBlackBox") 
        print("- Adjust scale factor (currently +10.5) if needed for your use case")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()