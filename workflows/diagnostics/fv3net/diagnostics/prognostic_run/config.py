from typing import Mapping, Sequence
import intake


def get_verification_entries(
    name: str, catalog: intake.catalog.Catalog
) -> Mapping[str, Sequence[str]]:
    """Given simulation name, return catalog keys for c48 dycore and physics data.
    
    Args:
        name: Simulation to use for verification.
        catalog: Catalog to search for verification data.
        
    Returns:
        Mapping from category name ('physics', 'dycore', or '3d') to sequence
        of catalog keys representing given diagnostics for specified simulation.
    """
    entries = {"2d": [], "3d": []}
    for item in catalog:
        metadata = catalog[item].metadata
        item_simulation = metadata.get("simulation", None)
        item_grid = metadata.get("grid", None)
        item_category = metadata.get("category", None)

        if item_simulation == name and item_grid == "c48":
            if item_category is not None:
                entries[item_category].append(item)

    if len(entries["2d"]) == 0:
        raise ValueError(
            f"No c48 2d diagnostics found in catalog for simulation {name}."
        )

    return entries
