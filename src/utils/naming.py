"""Naming conventions and clip ID parsing utilities."""


def clip_id_to_patient(clip_id):
    """Extract patient ID from clip ID.

    Examples:
        'kku_w_01_FV' -> 'kku'
        'ajy_cr_02_LV' -> 'ajy'
        'hdi_c_s_01_RV' -> 'hdi'
        'hdi_cc_s_01_FV' -> 'hdi'
        'hdi_s_cc_01_FV' -> 'hdi'
    """
    parts = clip_id.split("_")
    simple_codes = {"w", "cr", "sr"}
    compound_codes = {"c_s", "s_c", "cc_s", "s_cc"}

    for i, part in enumerate(parts):
        # Check 2-part compound codes first (higher priority)
        if i + 1 < len(parts):
            compound = f"{part}_{parts[i + 1]}"
            if compound in compound_codes:
                return "_".join(parts[:i])
        # Then check simple codes
        if part in simple_codes:
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


def clip_id_to_triplet_base(clip_id):
    """Strip view suffix to get triplet base identifier.

    Examples:
        'kku_w_01_FV' -> 'kku_w_01'
        'ly_cr_02_LV' -> 'ly_cr_02'
        'hdi_c_s_01_RV' -> 'hdi_c_s_01'
    """
    for suffix in ("_FV", "_LV", "_RV"):
        if clip_id.endswith(suffix):
            return clip_id[: -len(suffix)]
    return clip_id


def clip_id_to_movement(clip_id):
    """Extract movement code from clip ID.

    Returns:
        Movement code string (e.g., 'w', 'cr', 'c_s', 's_c', 'sr',
        'cc_s', 's_cc').
    """
    parts = clip_id.split("_")
    movement_codes = {"w", "cr", "sr"}
    compound_codes = {"c_s", "s_c", "cc_s", "s_cc"}

    for i, part in enumerate(parts):
        if part in movement_codes:
            return part
        # Check for compound movement codes (c_s, s_c, cc_s, s_cc)
        if i + 1 < len(parts):
            compound = f"{part}_{parts[i + 1]}"
            if compound in compound_codes:
                return compound
    return None
