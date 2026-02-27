# MuTang++ Notes

## Correlated Single-Position Mutations

**Goal:** Establish correlated single-position mutations.

- For each position-wise mutation proposition, compare similarity to mutations proposed at other positions
- This yields a ~525 x 525 similarity matrix; mark entries we do not want to compare (i.e. mutations from the same position)
- From this, derive a graph with similarity-weighted edges
- Threshold similarities to obtain different connected components
  - Constraint: connected components cannot contain mutations at the same position
- Alternative framing: clustering (dendrograms + some metric) to obtain correlated position-wise mutations

## Mutation Potentials

- Derive potentials over mutations from the similarity matrix
- How to make more probable mutations when there is correlation at different positions?
- Put Karol's things within this framework

## Mutation Representation in Ambient Space

- How is a mutation represented in the ambient space?
  - The softmax logits increase for the given direction while the current position decreases (or: other positions decrease and one rises, which would make less sense)
  - Apply on the log-softmax?
- Consider the basis of mutations in the ambient space
- **Q:** If we sample directions on the tangent space, what will be the distribution of the derivatives for logits?

## Perplexity & Distance Comparisons

- **Q:** How can we interpret the mutant's perplexity (under parent) as mutation potentials?
  - See `hydramp.decoder_forward`
- Compare distances from two methods to the parent
- Compare perplexity to check if we are not getting out of distribution:
  1. Conditioned on parent
  2. Conditioned on mutant embedding


## Downstream evaluation
- one should check how sampling from the decoder distribution works