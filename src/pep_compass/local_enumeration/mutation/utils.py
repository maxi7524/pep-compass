from collections import defaultdict

import numpy as np



def get_mutations_from_s_u_standard(
    s: np.ndarray,
    u: np.ndarray,
    max_len: int,
    alphabet_size: int,
    direction_significance_threshold: float,
    min_number_of_directions: int,
    token_threshold: float,
) -> dict[int, list[int]]:
    """Compute possible amino acid mutations from SVD tensors S and U.

    Given the singular values (S) and left singular vectors (U) from the SVD
    of the decoder Jacobian, determine which amino acid substitutions are
    significant at each sequence position.

    Args:
        s: Array of singular values (significance scores) for directions.
            Shape: ``[n_directions]``.
        u: Left singular vectors (tangent-space directions).
            Shape: ``[max_len * alphabet_size, n_directions]``.
        max_len: Maximum allowed peptide length.
        alphabet_size: Size of the amino acid alphabet (including padding token).
        direction_significance_threshold: Minimum singular value to consider a
            direction as significant.
        min_number_of_directions: Minimum number of directions to evaluate,
            even if fewer exceed the significance threshold.
        token_threshold: Minimum absolute weight for a token at the selected
            position to be included as a candidate mutation.

    Returns:
        A dictionary mapping sequence positions to lists of amino acid indices
        that are candidate mutations at that position.
    """
    assert isinstance(s, np.ndarray) and isinstance(u, np.ndarray), ValueError(
        f"s and u should be numpy arrays, got {type(s)} and {type(u)} instead."
    )
    assert s.ndim == 1, ValueError(f"s should be 1D, got {s.ndim}D instead.")
    assert u.ndim == 2, ValueError(f"u should be 2D, got {u.ndim}D instead.")

    number_of_directions = max(
        int((s > direction_significance_threshold).sum()),
        min_number_of_directions,
    )

    mutations: dict[int, list[int]] = defaultdict(list)
    for direction_nb in range(number_of_directions):
        current_table = np.abs(
            u[:, direction_nb].reshape((max_len, alphabet_size))
        )
        change_position = int(current_table.sum(axis=1).argmax())

        for j in range(1, current_table.shape[1]):
            if current_table[change_position, j] > token_threshold:
                mutations[change_position].append(j)

    return mutations
