"""Naming conventions and clip ID parsing utilities.

Supports two naming conventions:
  Old: {patient}_{movement}_{num}_{VIEW}  e.g. kku_w_01_FV
  New: {patient}_{view}_{movement}_{num}  e.g. kcw_f_sr_01

View code mapping:  f -> FV,  l -> LV,  r -> RV
"""

# Short view codes used in the new naming convention
_NEW_VIEW = {"f": "FV", "l": "LV", "r": "RV"}


def _is_new_convention(parts):
    """Check if clip ID parts follow the new naming convention.

    New convention has a single-letter view code at index 1.
    """
    return len(parts) >= 3 and parts[1] in _NEW_VIEW


def clip_id_to_patient(clip_id):
    """Extract patient ID from clip ID.

    Examples:
        'kku_w_01_FV' -> 'kku'          (old convention)
        'hdi_c_s_01_RV' -> 'hdi'        (old convention)
        'kcw_f_sr_01' -> 'kcw'          (new convention)
        'hja_l_c_s_01' -> 'hja'         (new convention)
    """
    parts = clip_id.split("_")

    # New convention: view code at index 1 → patient is parts[0]
    if _is_new_convention(parts):
        return parts[0]

    # Old convention: scan for movement code
    simple_codes = {"w", "cr", "sr"}
    compound_codes = {"c_s", "s_c", "cc_s", "s_cc"}

    for i, part in enumerate(parts):
        if i + 1 < len(parts):
            compound = f"{part}_{parts[i + 1]}"
            if compound in compound_codes:
                return "_".join(parts[:i])
        if part in simple_codes:
            return "_".join(parts[:i])
    return parts[0]


def clip_id_to_view(clip_id):
    """Extract camera view from clip ID.

    Returns:
        'FV', 'LV', or 'RV', or None if not found.

    Examples:
        'kku_w_01_FV' -> 'FV'           (old convention)
        'kcw_f_sr_01' -> 'FV'           (new convention)
        'hja_l_c_s_01' -> 'LV'         (new convention)
    """
    parts = clip_id.split("_")

    # New convention: view code at index 1
    if _is_new_convention(parts):
        return _NEW_VIEW[parts[1]]

    # Old convention: view suffix at end
    for suffix in ["FV", "LV", "RV"]:
        if clip_id.endswith(suffix):
            return suffix
    return None


def clip_id_to_triplet_base(clip_id):
    """Strip view identifier to get triplet base.

    Examples:
        'kku_w_01_FV' -> 'kku_w_01'     (old convention)
        'hdi_c_s_01_RV' -> 'hdi_c_s_01' (old convention)
        'kcw_f_sr_01' -> 'kcw_sr_01'    (new convention)
        'hja_l_c_s_01' -> 'hja_c_s_01'  (new convention)
    """
    parts = clip_id.split("_")

    # New convention: remove view code at index 1
    if _is_new_convention(parts):
        return "_".join([parts[0]] + parts[2:])

    # Old convention: strip _FV/_LV/_RV suffix
    for suffix in ("_FV", "_LV", "_RV"):
        if clip_id.endswith(suffix):
            return clip_id[: -len(suffix)]
    return clip_id


def clip_id_to_movement(clip_id):
    """Extract movement code from clip ID.

    Returns:
        Movement code string (e.g., 'w', 'cr', 'c_s', 's_c', 'sr',
        'cc_s', 's_cc').

    Examples:
        'kku_w_01_FV' -> 'w'            (old convention)
        'hdi_c_s_01_RV' -> 'c_s'        (old convention)
        'kcw_f_sr_01' -> 'sr'           (new convention)
        'hja_l_c_s_01' -> 'c_s'         (new convention)
    """
    parts = clip_id.split("_")

    # New convention: strip view code before scanning
    if _is_new_convention(parts):
        parts = [parts[0]] + parts[2:]

    movement_codes = {"w", "cr", "sr"}
    compound_codes = {"c_s", "s_c", "cc_s", "s_cc"}

    for i, part in enumerate(parts):
        if i + 1 < len(parts):
            compound = f"{part}_{parts[i + 1]}"
            if compound in compound_codes:
                return compound
        if part in movement_codes:
            return part
    return None


def triplet_base_to_front_clip_id(triplet_base):
    """Convert triplet base to the front-view clip ID (new naming convention).

    Examples:
        'hja_c_s_01' -> 'hja_f_c_s_01'
        'kcw_sr_01'  -> 'kcw_f_sr_01'
    """
    parts = triplet_base.split("_")
    return parts[0] + "_f_" + "_".join(parts[1:])
