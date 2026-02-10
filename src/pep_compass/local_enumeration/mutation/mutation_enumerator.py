from collections import defaultdict
from copy import deepcopy
import numpy as np
from einops import rearrange

from pep_compass.local_enumeration.mutation.utils import get_mutations_from_s_u


class MutationEnumerator:
    """Base class for peptide mutation enumerators."""

    def mutate(self, sequence: str, **kwargs) -> set[str]:
        """Enumerate mutations for a given sequence.

        Args:
            sequence: The peptide sequence to mutate.
            **kwargs: Additional arguments required by specific implementations.

        Returns:
            A list of mutated peptide sequences.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


class NoMutationEnumerator(MutationEnumerator):
    """Enumerator that returns the peptide unchanged."""

    def mutate(self, peptide: str, **kwargs) -> set[str]:
        """Return the input peptide without mutations.

        Args:
            peptide: The input peptide sequence.
            **kwargs: Unused keyword arguments.

        Returns:
            A single-element set containing the original peptide.
        """
        return {peptide}


class MutationEnumerationInTangentSpace(MutationEnumerator):
    """Enumerate peptide mutations based on directions in tangent space.

    Attributes:
        max_len: Maximum allowed peptide length.
        direction_significance_threshold: Minimum significance threshold for direction selection.
        min_number_of_directions: Minimum number of mutation directions to consider.
        token_threshold: Threshold for mutation probability per position.
        alphabet: List of sequence tokens (including a space for padding).
    """

    def __init__(
        self,
        max_len: int = 25,
        direction_significance_threshold: float = 1e-3,
        min_number_of_directions: int = 5,
        token_threshold: float = 0.1,
        alphabet: list[str] | None = None,
    ):
        super().__init__()
        self.max_len = max_len
        self.direction_significance_threshold = direction_significance_threshold
        self.min_number_of_directions = min_number_of_directions
        self.token_threshold = token_threshold
        self.alphabet = alphabet or list(" ACDEFGHIKLMNPQRSTVWY")

    def mutate(
        self, peptide: str, S: np.ndarray, U: np.ndarray, **kwargs: dict
    ) -> set[str]:
        """Generate mutated peptides using tangent-space directions.

        Args:
            peptide: The input peptide sequence.
            S: Direction significance tensor (numpy array).
            U: Direction basis tensor (numpy array).

        Returns:
            A set of unique mutated peptide sequences.
        """

        mutations = self.get_mutations_from_s_u(S, U)

        mutated_peptides = set(self.mutate_peptide(peptide, mutations))

        return mutated_peptides

    def get_mutations_from_s_u(
        self,
        s: np.ndarray,
        u: np.ndarray,
    ) -> dict[int, list[int]]:
        """Compute possible amino acid mutations from tensors S and U.

        This delegates to the shared :func:`get_mutations_from_s_u` utility.
        """
        return get_mutations_from_s_u(
            s=s,
            u=u,
            max_len=self.max_len,
            alphabet_size=len(self.alphabet),
            direction_significance_threshold=self.direction_significance_threshold,
            min_number_of_directions=self.min_number_of_directions,
            token_threshold=self.token_threshold,
        )

    def aux_mutate(
        self,
        current_sequence: str,
        mutations: dict[int, set[int]],
        agg: list[str],
    ) -> None:
        """Recursively enumerate all mutations up to the maximum length.

        Args:
            current_sequence: The sequence built so far.
            mutations: Dictionary mapping positions to possible amino acid indices.
            agg: Aggregator list that stores complete mutated sequences.
        """
        current_pos = len(current_sequence)
        if current_pos == self.max_len:
            agg.append(current_sequence.replace(" ", ""))
            return

        for new_aa in mutations[current_pos]:
            self.aux_mutate(current_sequence + self.alphabet[new_aa], mutations, agg)

    def mutate_peptide(
        self,
        peptide: str,
        mutations: dict[int, list[int]],
    ) -> list[str]:
        """Apply mutations to a peptide sequence.

        Args:
            peptide: The input peptide sequence.
            mutations: A dictionary of candidate mutations by position.

        Returns:
            A list of mutated peptide sequences.
        """
        padded_peptide = peptide + "".join([" "] * (self.max_len - len(peptide)))
        new_mutations = deepcopy(mutations)

        for aa_pos, aa in enumerate(padded_peptide):
            new_mutations[aa_pos].append(self.alphabet.index(aa))
            new_mutations[aa_pos] = set(new_mutations[aa_pos])

        result_list = []
        self.aux_mutate("", new_mutations, result_list)
        return result_list


class AblationRandomMutationEnumerator(MutationEnumerationInTangentSpace):
    """Ablation study variant that introduces random amino acid substitutions."""

    def __init__(
        self,
        max_len: int = 25,
        direction_significance_threshold: float = 1e-3,
        min_number_of_directions: int = 5,
        token_threshold: float = 0.1,
        alphabet: list[str] | None = None,
    ):
        super().__init__(
            max_len=max_len,
            direction_significance_threshold=direction_significance_threshold,
            min_number_of_directions=min_number_of_directions,
            token_threshold=token_threshold,
            alphabet=alphabet,
        )

    def get_mutations_from_s_u(
        self,
        s: np.ndarray,
        u: np.ndarray,
    ) -> dict[int, list[int]]:
        """Generate random mutations for each selected direction.

        Args:
            s: Array of significance scores for directions.
            u: Tangent-space directions (unused beyond shape context).

        Returns:
            A dictionary mapping sequence positions to lists of random amino acid indices.
        """
        number_of_directions = max(
            int((s > self.direction_significance_threshold).sum()),
            self.min_number_of_directions,
        )

        mutations = defaultdict(list)
        for direction_idx in range(number_of_directions):
            current_table = np.abs(
                u[:, direction_idx].reshape((self.max_len, len(self.alphabet)))
            )
            change_position = current_table.sum(axis=1).argmax()

            for j in range(1, current_table.shape[1]):
                if current_table[change_position, j] > self.token_threshold:
                    random_amino_acid = np.random.randint(0, len(self.alphabet))
                    mutations[change_position].append(random_amino_acid)

        return mutations
