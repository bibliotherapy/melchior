"""Patient vs caregiver identification from multi-person pose detections.

Uses height ratio heuristic: the adult (caregiver) skeleton is >= 1.3x
the child (patient) skeleton height. Falls back to center-of-frame position
when height ratio is ambiguous.
"""
