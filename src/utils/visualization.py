"""
FencerAI Visualization Utilities
==============================
Version: 1.0 | Last Updated: 2026-03-28

Visualization utilities for skeleton overlay and feature display.
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import cv2
import numpy as np

from src.utils.schemas import FencerPose, Keypoint
from src.utils.constants import COCO_SKELETON_CONNECTIONS


# =============================================================================
# Colors
# =============================================================================

FENCER_COLORS = {
    0: (255, 100, 100),  # Left fencer - Red
    1: (100, 100, 255),  # Right fencer - Blue
}

SKELETON_COLOR = (200, 200, 200)  # Gray
KEYPOINT_COLOR = (0, 255, 0)  # Green
KEYPOINT_REJECT_COLOR = (0, 0, 255)  # Red (low confidence)
BBOX_COLOR = (255, 255, 0)  # Yellow


# =============================================================================
# Skeleton Drawing
# =============================================================================

def draw_keypoints(
    frame: np.ndarray,
    keypoints: List[Keypoint],
    color: Tuple[int, int, int] = KEYPOINT_COLOR,
    min_conf: float = 0.3,
    radius: int = 3,
    thickness: int = -1,  # -1 = filled
) -> np.ndarray:
    """
    Draw keypoints on frame.

    Args:
        frame: Input frame (H, W, 3) in BGR
        keypoints: List of Keypoint objects
        color: Color for high-confidence keypoints
        min_conf: Minimum confidence to draw
        radius: Keypoint circle radius
        thickness: Line thickness (-1 for filled)

    Returns:
        Frame with keypoints drawn
    """
    for kp in keypoints:
        if kp.conf >= min_conf:
            cv2.circle(
                frame,
                (int(kp.x), int(kp.y)),
                radius,
                color,
                thickness,
            )
        else:
            cv2.circle(
                frame,
                (int(kp.x), int(kp.y)),
                radius,
                KEYPOINT_REJECT_COLOR,
                thickness,
            )
    return frame


def draw_skeleton(
    frame: np.ndarray,
    keypoints: List[Keypoint],
    color: Tuple[int, int, int] = SKELETON_COLOR,
    min_conf: float = 0.3,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw skeleton connections on frame.

    Args:
        frame: Input frame (H, W, 3) in BGR
        keypoints: List of Keypoint objects
        color: Color for skeleton lines
        min_conf: Minimum confidence for both endpoints
        thickness: Line thickness

    Returns:
        Frame with skeleton drawn
    """
    for connection in COCO_SKELETON_CONNECTIONS:
        kp1_idx, kp2_idx = connection

        if kp1_idx >= len(keypoints) or kp2_idx >= len(keypoints):
            continue

        kp1 = keypoints[kp1_idx]
        kp2 = keypoints[kp2_idx]

        # Only draw if both keypoints have sufficient confidence
        if kp1.conf >= min_conf and kp2.conf >= min_conf:
            cv2.line(
                frame,
                (int(kp1.x), int(kp1.y)),
                (int(kp2.x), int(kp2.y)),
                color,
                thickness,
            )

    return frame


def draw_bbox(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    color: Tuple[int, int, int] = BBOX_COLOR,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding box on frame.

    Args:
        frame: Input frame (H, W, 3) in BGR
        bbox: Bounding box [x1, y1, x2, y2]
        color: Color for bbox
        thickness: Line thickness

    Returns:
        Frame with bbox drawn
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame


def draw_fencer_id(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    fencer_id: int,
    color: Tuple[int, int, int],
) -> np.ndarray:
    """
    Draw fencer ID label above bbox.

    Args:
        frame: Input frame (H, W, 3) in BGR
        bbox: Bounding box [x1, y1, x2, y2]
        fencer_id: 0 = Left, 1 = Right
        color: Color for label

    Returns:
        Frame with label drawn
    """
    x1, y1, _, _ = [int(v) for v in bbox]
    label = f"Fencer {fencer_id}"
    cv2.putText(
        frame,
        label,
        (x1, max(0, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )
    return frame


def draw_pose(
    frame: np.ndarray,
    pose: FencerPose,
    fencer_color: Tuple[int, int, int],
    min_conf: float = 0.3,
) -> np.ndarray:
    """
    Draw complete pose (skeleton + keypoints + bbox + label).

    Args:
        frame: Input frame (H, W, 3) in BGR
        pose: FencerPose to draw
        fencer_color: Color for this fencer
        min_conf: Minimum confidence threshold

    Returns:
        Frame with pose drawn
    """
    # Draw skeleton connections
    frame = draw_skeleton(frame, pose.keypoints, fencer_color, min_conf)

    # Draw keypoints
    frame = draw_keypoints(frame, pose.keypoints, fencer_color, min_conf)

    # Draw bbox
    frame = draw_bbox(frame, pose.bbox, fencer_color)

    # Draw fencer ID
    frame = draw_fencer_id(frame, pose.bbox, pose.fencer_id, fencer_color)

    return frame


def draw_frame_overlay(
    frame: np.ndarray,
    poses: List[FencerPose],
    min_conf: float = 0.3,
) -> np.ndarray:
    """
    Draw all poses on frame with color coding.

    Args:
        frame: Input frame (H, W, 3) in BGR
        poses: List of FencerPose objects
        min_conf: Minimum confidence threshold

    Returns:
        Frame with all poses drawn
    """
    for pose in poses:
        color = FENCER_COLORS.get(pose.fencer_id, SKELETON_COLOR)
        frame = draw_pose(frame, pose, color, min_conf)

    return frame


# =============================================================================
# Info Overlay
# =============================================================================

def draw_info_overlay(
    frame: np.ndarray,
    frame_id: int,
    fps: float,
    n_fencers: int,
) -> np.ndarray:
    """
    Draw processing info overlay on frame.

    Args:
        frame: Input frame (H, W, 3) in BGR
        frame_id: Current frame ID
        fps: Processing fps
        n_fencers: Number of fencers detected

    Returns:
        Frame with info overlay
    """
    # Background rectangle
    h, w = frame.shape[:2]
    overlay_h = 60
    cv2.rectangle(frame, (0, 0), (w, overlay_h), (0, 0, 0), -1)
    cv2.addWeighted(frame[0:overlay_h, 0:w], 0.6, frame[0:overlay_h, 0:w], 0.4, 0, frame[0:overlay_h, 0:w])

    # Text
    info_text = f"Frame: {frame_id} | FPS: {fps:.1f} | Fencers: {n_fencers}"
    cv2.putText(
        frame,
        info_text,
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    return frame


# =============================================================================
# Feature Matrix Heatmap Export
# =============================================================================

def create_feature_heatmap(
    features: np.ndarray,
    fencer_id: int = 0,
    normalize: bool = True,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Create a heatmap visualization of feature matrix for one fencer.

    Args:
        features: Feature matrix of shape (N_frames, 101) for one fencer
        fencer_id: 0 = Left (Red), 1 = Right (Blue) - used for coloring
        normalize: Whether to normalize features to [0, 1]
        colormap: OpenCV colormap to use

    Returns:
        Heatmap image of shape (H, W, 3) in BGR
    """
    n_frames, n_features = features.shape

    # Flatten for visualization
    heatmap_data = features.T  # Shape: (101, N_frames)

    # Normalize if requested
    if normalize:
        min_val = np.min(heatmap_data)
        max_val = np.max(heatmap_data)
        if max_val - min_val > 1e-6:
            heatmap_data = (heatmap_data - min_val) / (max_val - min_val)
        else:
            heatmap_data = np.zeros_like(heatmap_data)

    # Scale to reasonable size
    height_per_feature = 20
    width_per_frame = 2

    h = n_features * height_per_feature
    w = n_frames * width_per_frame

    # Resize to target dimensions
    heatmap_resized = cv2.resize(heatmap_data, (w, h), interpolation=cv2.INTER_LINEAR)

    # Apply colormap (jet: blue=low, red=high)
    heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), colormap)

    # Add color bar on the right using jet colormap
    colorbar_width = 50
    colorbar = np.zeros((h, colorbar_width, 3), dtype=np.uint8)
    for i in range(h):
        val = int(255 * (1 - i / h))
        colorbar[i, :] = [val, val, val]  # BGR grayscale gradient

    # Add feature region labels
    labeled_heatmap = np.zeros((h + 60, w + colorbar_width + 20, 3), dtype=np.uint8)
    labeled_heatmap[:] = 30  # Dark background

    # Place heatmap
    labeled_heatmap[30:30+h, 10:10+w] = heatmap_colored

    # Place colorbar
    labeled_heatmap[30:30+h, 10+w:10+w+colorbar_width] = colorbar

    # Add labels
    cv2.putText(labeled_heatmap, f"Fencer {fencer_id} Features", (10, 20),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Add axis labels
    cv2.putText(labeled_heatmap, "Time →", (w//2 - 30, h + 50),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Feature region markers
    feature_regions = [
        (0, 24, "Geometry"),
        (24, 26, "CoM"),
        (26, 37, "Distance"),
        (37, 43, "Angular"),
        (43, 49, "Arm"),
        (49, 73, "Velocity"),
        (73, 97, "Accel"),
        (97, 101, "Meta"),
    ]

    for start, end, name in feature_regions:
        y_pos = 30 + start * height_per_feature
        cv2.putText(labeled_heatmap, name, (5, y_pos + 15),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

    return labeled_heatmap


def export_feature_heatmap(
    feature_matrix: np.ndarray,
    output_path: str,
) -> None:
    """
    Export feature matrix as heatmap images.

    Args:
        feature_matrix: Feature matrix of shape (N_frames, 2, 101)
        output_path: Path to save heatmap images (without extension)
    """
    n_frames, n_fencers, n_features = feature_matrix.shape

    # Create heatmap for each fencer
    for fencer_id in range(n_fencers):
        features = feature_matrix[:, fencer_id, :]  # Shape: (N_frames, 101)
        heatmap = create_feature_heatmap(features, fencer_id=fencer_id)

        # Save
        heatmap_path = f"{output_path}_fencer{fencer_id}_heatmap.png"
        cv2.imwrite(heatmap_path, heatmap)

    # Create combined heatmap
    combined_h = 30 + 101 * 20 + 60
    combined_w = n_frames * 2 * 2 + 50 + 20  # Two fencers side by side

    combined = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
    combined[:] = 30

    # Left fencer (top)
    left_heatmap = create_feature_heatmap(feature_matrix[:, 0, :], fencer_id=0)
    h, w = left_heatmap.shape[:2]
    combined[0:h, 0:w] = left_heatmap

    # Right fencer (bottom, offset)
    right_heatmap = create_feature_heatmap(feature_matrix[:, 1, :], fencer_id=1)
    offset_y = h + 10
    combined[offset_y:offset_y + h, 0:w] = right_heatmap

    # Save combined
    combined_path = f"{output_path}_combined_heatmap.png"
    cv2.imwrite(combined_path, combined)

    # Also save as numpy for further processing
    npy_path = f"{output_path}_heatmap_data.npy"
    np.save(npy_path, feature_matrix)
