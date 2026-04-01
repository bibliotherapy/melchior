"""Layer 3: Extended metadata / assistive context vector encoder.

Encodes the 18D per-patient context vector from assistive_annotations.json,
extending the original 7D metadata (sex, age, movement statuses) with
walker type, AFO presence, and assistance levels per movement.
"""
