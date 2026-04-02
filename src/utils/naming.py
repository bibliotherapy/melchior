"""Naming conventions and clip ID parsing utilities."""


def clip_id_to_patient(clip_id):
    """Extract patient ID from clip ID.

    Examples:
        'kku_w_01_FV' -> 'kku'
        'ajy_cr_02_LV' -> 'ajy'
        'hdi_c_s_01_RV' -> 'hdi'
    """
    parts = clip_id.split("_")
    movement_codes = {"w", "cr", "c", "s", "sr"}
    for i, part in enumerate(parts):
        if part in movement_codes:
            return "_".join(parts[:i])
    return parts[0]


def clip_id_to_view(clip_id):
    """Extract camera view from clip ID.

    Returns:
        'FV', 'LV', or 'RV', or None if not found.
    """
    for suffix in ["FV", "LV", "RV"]:
        if clip_id.endswith(suffix):
            return suffix
    return None


def clip_id_to_movement(clip_id):
    """Extract movement code from clip ID.

    Returns:
        Movement code string (e.g., 'w', 'cr', 'c_s', 's_c', 'sr').
    """
    parts = clip_id.split("_")
    movement_codes = {"w", "cr", "sr"}
    compound_codes = {"c_s", "s_c"}

    for i, part in enumerate(parts):
        if part in movement_codes:
            return part
        # Check for compound movement codes (c_s, s_c)
        if i + 1 < len(parts):
            compound = f"{part}_{parts[i + 1]}"
            if compound in compound_codes:
                return compound
    return None
