"""Traverse executable objects without changing their lifecycle or RNG state."""


def iter_steps(root):
    """Yield composite operations and leaf components in structural order."""
    yield root
    children = list(getattr(root, "steps", ()))
    if hasattr(root, "body"):
        children.append(root.body)
    children.extend(getattr(root, "branches", {}).values())
    if hasattr(root, "walker"):
        children.extend((root.walker, root.mutation_generator, root.filters))
    for child in children:
        yield from iter_steps(child)


def find_step(root, node_id):
    """Find an executable by its stable declaration identity."""
    return next(
        (
            step
            for step in iter_steps(root)
            if getattr(step, "node_id", None) == node_id
        ),
        None,
    )
