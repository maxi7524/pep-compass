#!/usr/bin/env python3
"""Calculate ranks for peptide sequences based on HydrAMP latent space SVD distribution."""

import torch
import numpy as np
import pandas as pd
from typing import List, Union, Literal
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
import os
from datetime import datetime

# Add the source directory to Python path to use local code instead of installed package
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent.parent  # Go up 3 levels to project root
src_dir = project_root / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

try:
    from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import HydrAMPEncoderDecoder
except ImportError as e:
    print(f"Error importing HydrAMPEncoderDecoder: {e}")
    print(f"Make sure you're running this script from the pep-compass project directory.")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project root detected as: {project_root}")
    print(f"Source directory: {src_dir}")
    sys.exit(1)


def calculate_ranks_for_multiple_kappas(
    peptides: List[str], 
    kappas: List[float],
    device: str = "cpu"
) -> pd.DataFrame:
    """
    Calculate ranks for multiple kappa values efficiently.
    
    Args:
        peptides: List of peptide sequences
        kappas: List of kappa threshold parameters
        device: Device to run computations on
        
    Returns:
        DataFrame with peptides and ranks for each kappa
    """
    # Check if weights exist before initializing model
    weights_check_paths = [
        project_root / "src" / "pep_compass" / "models" / "hydramp" / "weights",
        Path("src/pep_compass/models/hydramp/weights"),
        Path("results/weights/hydramp"),  # Alternative location
    ]
    
    weights_found = False
    for weights_path in weights_check_paths:
        if weights_path.exists() and (weights_path / "encoder_weights.pickle").exists():
            print(f"Found HydrAMP weights at: {weights_path}")
            weights_found = True
            break
    
    if not weights_found:
        print("ERROR: HydrAMP weights not found!")
        print("Searched in the following locations:")
        for p in weights_check_paths:
            print(f"  - {p}")
        print("\nPlease ensure the HydrAMP model weights are available.")
        return pd.DataFrame()
    
    # Validate device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"Warning: CUDA device '{device}' requested but CUDA is not available. Using CPU instead.")
        device = "cpu"
    
    # Initialize HydrAMP model
    try:
        hydramp = HydrAMPEncoderDecoder(
            jacobian_mode="approx",
            device=device,
            jacobian_eps=1e-6,
            field_eps=1e-6
        )
        print(f"Successfully initialized HydrAMP model on device: {device}")
    except Exception as e:
        print(f"Error initializing HydrAMP model: {e}")
        return pd.DataFrame()
    
    # Calculate SVD for all peptides once
    all_s_squared = []
    valid_peptides = []
    
    for peptide in tqdm(peptides, desc="Computing SVD for peptides"):
        try:
            # Encode peptide to latent space
            z = hydramp.encode_peptides([peptide])
            
            # Calculate Jacobian of decoder at latent point
            jac = hydramp.decoder_jacobian(z)
            
            # Perform SVD decomposition
            U, S, Vt = torch.linalg.svd(jac, full_matrices=False)
            
            # Calculate squared singular values
            s_squared = (S[0] ** 2).detach().cpu().numpy()
            all_s_squared.append(s_squared)
            valid_peptides.append(peptide)
            
        except Exception as e:
            print(f"Error processing peptide {peptide}: {e}")
    
    # Now calculate ranks for all kappa values in wide format
    results = []
    for peptide, s_squared in zip(valid_peptides, all_s_squared):
        row = {'sequence': peptide}
        for kappa in kappas:
            rank = int(np.sum(s_squared > kappa))
            row[f"{kappa:.0e}"] = rank
        results.append(row)
    
    return pd.DataFrame(results)


def calculate_rank_for_peptides(
    peptides: List[str], 
    kappa: float = 0.01,
    device: str = "cpu"
) -> List[int]:
    """
    Calculate rank for a list of peptides based on HydrAMP latent space SVD.
    
    The rank is defined as the number of squared singular values that exceed 
    the kappa parameter after SVD decomposition of the decoder Jacobian.
    
    Args:
        peptides: List of peptide sequences
        kappa: Threshold parameter for singular value selection
        device: Device to run computations on ("cpu" or "cuda")
        
    Returns:
        List of ranks (integers) for each peptide
    """
    # Check if weights exist before initializing model
    weights_check_paths = [
        project_root / "src" / "pep_compass" / "models" / "hydramp" / "weights",
        Path("src/pep_compass/models/hydramp/weights"),
        Path("results/weights/hydramp"),  # Alternative location
    ]
    
    weights_found = False
    for weights_path in weights_check_paths:
        if weights_path.exists() and (weights_path / "encoder_weights.pickle").exists():
            print(f"Found HydrAMP weights at: {weights_path}")
            weights_found = True
            break
    
    if not weights_found:
        print("ERROR: HydrAMP weights not found!")
        print("Searched in the following locations:")
        for p in weights_check_paths:
            print(f"  - {p}")
        print("\nPlease ensure the HydrAMP model weights are available.")
        print("You may need to download them or check the README for setup instructions.")
        return [0] * len(peptides)
    
    # Validate device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"Warning: CUDA device '{device}' requested but CUDA is not available. Using CPU instead.")
        device = "cpu"
    
    # Initialize HydrAMP model
    try:
        hydramp = HydrAMPEncoderDecoder(
            jacobian_mode="approx",
            device=device,
            jacobian_eps=1e-6,
            field_eps=1e-6
        )
        print(f"Successfully initialized HydrAMP model on device: {device}")
    except Exception as e:
        print(f"Error initializing HydrAMP model: {e}")
        return [0] * len(peptides)
    
    ranks = []
    
    for peptide in tqdm(peptides, desc="Calculating ranks"):
        try:
            # Encode peptide to latent space
            z = hydramp.encode_peptides([peptide])
            
            # Calculate Jacobian of decoder at latent point
            jac = hydramp.decoder_jacobian(z)
            
            # Perform SVD decomposition
            U, S, Vt = torch.linalg.svd(jac, full_matrices=False)
            
            # Calculate squared singular values
            s_squared = (S[0] ** 2).detach().cpu().numpy()
            
            # Count how many exceed kappa threshold
            rank = int(np.sum(s_squared > kappa))
            ranks.append(rank)
            
        except Exception as e:
            print(f"Error processing peptide {peptide}: {e}")
            ranks.append(0)  # Default rank if error occurs
    
    return ranks


def calculate_rank_from_csv(
    csv_path: Union[str, Path],
    sequence_column: str = "sequence",
    kappa: float = 0.01,
    device: str = "cpu",
    output_path: Union[str, Path, None] = None
) -> pd.DataFrame:
    """
    Calculate ranks for peptides from a CSV file.
    
    Args:
        csv_path: Path to CSV file containing peptide sequences
        sequence_column: Name of column containing sequences
        kappa: Threshold parameter for singular value selection
        device: Device to run computations on
        output_path: Optional path to save results. If None, returns DataFrame only.
        
    Returns:
        DataFrame with original data plus 'rank' and 'kappa' columns
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    if sequence_column not in df.columns:
        raise ValueError(f"Column '{sequence_column}' not found in CSV. Available columns: {list(df.columns)}")
    
    # Extract peptides
    peptides = df[sequence_column].tolist()
    
    # Calculate ranks
    ranks = calculate_rank_for_peptides(peptides, kappa=kappa, device=device)
    
    # Add results to dataframe
    df['rank'] = ranks
    df['kappa'] = kappa
    
    # Save if output path provided
    if output_path is not None:
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
    
    return df


def calculate_svd_statistics(
    peptides: List[str],
    device: str = "cpu"
) -> pd.DataFrame:
    """
    Calculate detailed SVD statistics for peptides (for analysis purposes).
    
    Args:
        peptides: List of peptide sequences
        device: Device to run computations on
        
    Returns:
        DataFrame with detailed SVD statistics
    """
    try:
        hydramp = HydrAMPEncoderDecoder(
            jacobian_mode="approx", 
            device=device,
            jacobian_eps=1e-6,
            field_eps=1e-6
        )
    except Exception as e:
        print(f"Error initializing HydrAMP model: {e}")
        return pd.DataFrame()
    
    results = []
    
    for peptide in tqdm(peptides, desc="Computing SVD statistics"):
        try:
            # Encode and get SVD
            z = hydramp.encode_peptides([peptide])
            jac = hydramp.decoder_jacobian(z)
            U, S, Vt = torch.linalg.svd(jac, full_matrices=False)
            
            s_values = S[0].detach().cpu().numpy()
            s_squared = s_values ** 2
            
            # Calculate statistics
            result = {
                'peptide': peptide,
                'n_singular_values': len(s_values),
                'max_singular_value': float(s_values.max()),
                'min_singular_value': float(s_values.min()),
                'mean_singular_value': float(s_values.mean()),
                'std_singular_value': float(s_values.std()),
                'max_singular_value_squared': float(s_squared.max()),
                'mean_singular_value_squared': float(s_squared.mean()),
                'sum_singular_values_squared': float(s_squared.sum()),
            }
            
            # Add rank calculations for different kappa values
            for kappa in [0.001, 0.01, 0.1, 1.0]:
                result[f'rank_kappa_{kappa}'] = int(np.sum(s_squared > kappa))
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing peptide {peptide}: {e}")
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Calculate peptide ranks based on HydrAMP SVD")
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Input CSV file path or comma-separated peptide sequences")
    parser.add_argument("--sequence_column", "-s", type=str, default="sequence",
                       help="Column name containing sequences (for CSV input)")
    parser.add_argument("--kappa", "-k", type=str, default="0.01",
                       help="Kappa threshold parameter(s) - single value or comma-separated list")
    parser.add_argument("--output", "-o", type=str, 
                       help="Output CSV file path")
    parser.add_argument("--device", "-d", type=str, default="cpu",
                       help="Device to use (cpu/cuda)")
    parser.add_argument("--statistics", action="store_true",
                       help="Calculate detailed SVD statistics")
    
    args = parser.parse_args()
    
    # Parse kappa values
    kappa_values = [float(k.strip()) for k in args.kappa.split(',')]
    
    # Determine if input is file or peptide list
    if Path(args.input).exists():
        # Input is a file
        if args.statistics:
            df = pd.read_csv(args.input)
            peptides = df[args.sequence_column].tolist()
            results_df = calculate_svd_statistics(peptides, device=args.device)
        else:
            # Generate output filename if not provided
            if not args.output:
                input_name = Path(args.input).stem
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if len(kappa_values) == 1:
                    kappa_str = f"{kappa_values[0]:.2e}".replace(".", "_").replace("-", "neg")
                else:
                    kappa_str = "multi"
                output_path = f"kappa_{kappa_str}_ranks_{input_name}_{timestamp}.csv"
            else:
                output_path = args.output
            
            if len(kappa_values) == 1:
                results_df = calculate_rank_from_csv(
                    csv_path=args.input,
                    sequence_column=args.sequence_column,
                    kappa=kappa_values[0],
                    device=args.device,
                    output_path=output_path
                )
            else:
                # Load CSV and calculate for multiple kappas
                df = pd.read_csv(args.input)
                peptides = df[args.sequence_column].tolist()
                results_df = calculate_ranks_for_multiple_kappas(peptides, kappa_values, device=args.device)
                results_df.to_csv(output_path, index=False)
                print(f"Results saved to: {output_path}")
    else:
        # Input is peptide sequences
        peptides = args.input.split(',')
        
        if args.statistics:
            results_df = calculate_svd_statistics(peptides, device=args.device)
        else:
            if len(kappa_values) == 1:
                # Single kappa value
                ranks = calculate_rank_for_peptides(peptides, kappa=kappa_values[0], device=args.device)
                results_df = pd.DataFrame({
                    'sequence': peptides,
                    f"{kappa_values[0]:.0e}": ranks
                })
                kappa_str = f"{kappa_values[0]:.2e}".replace(".", "_").replace("-", "neg")
            else:
                # Multiple kappa values
                results_df = calculate_ranks_for_multiple_kappas(peptides, kappa_values, device=args.device)
                kappa_str = "multi"
        
        # Save results to CSV
        if args.output:
            output_path = args.output
        else:
            # Generate automatic filename with kappa value and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"kappa_{kappa_str}_ranks_{timestamp}.csv"
        
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
    
    # Print summary
    if not args.statistics:
        if len(kappa_values) == 1:
            unique_peptides = len(results_df)
            print(f"\nProcessed {unique_peptides} peptides")
            print(f"Kappa threshold: {kappa_values[0]}")
            kappa_col = f"{kappa_values[0]:.0e}"
            print(f"Mean rank: {results_df[kappa_col].mean():.2f}")
            print(f"Rank range: {results_df[kappa_col].min()} - {results_df[kappa_col].max()}")
        else:
            unique_peptides = len(results_df)
            print(f"\\nProcessed {unique_peptides} peptides with {len(kappa_values)} kappa values")
            print(f"Kappa thresholds: {kappa_values}")
            print(f"Overall rank statistics:")
            # Calculate statistics for each kappa column
            kappa_cols = [f"{kappa:.0e}" for kappa in kappa_values]
            stats_df = results_df[kappa_cols].agg(['mean', 'min', 'max']).round(2)
            print(stats_df)
    else:
        print(f"\nProcessed {len(results_df)} peptides with detailed SVD statistics")
    
    # Display first few results
    print(f"\nFirst few results:")
    print(results_df.head())


if __name__ == "__main__":
    main()
