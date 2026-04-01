"""3D triangulation for patient and caregiver skeletons.

Uses Direct Linear Transform (DLT) via OpenCV triangulatePoints to
reconstruct 3D joint positions from multi-view 2D detections.
Produces (T, 17, 3) arrays for both patient and caregiver.
"""
