"""
utils/severity.py
=================
Severity calculation for pothole detections.

IMPORTANT NOTICE:
The severity levels (Low / Medium / High) are project-defined categorizations
based on bounding-box area and detection confidence.
These are NOT official RDD2022 severity labels.
RDD2022 does NOT provide severity annotations.
Thresholds are configurable in config.yaml under the 'severity' section.
"""

from typing import Tuple
from utils.helpers import load_config, setup_logger

logger = setup_logger(__name__)


class SeverityCalculator:
    """
    Calculates pothole severity based on:
    1. Bounding-box area as a fraction of image area
    2. Detection confidence score
    
    Both factors are scored 0/1 and averaged to produce a final score:
      - score >= high_score  => High
      - score >= medium_score => Medium
      - else                  => Low
    
    Thresholds are loaded from config.yaml. This class can be instantiated
    with default thresholds if config is unavailable.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize with configuration.
        
        Args:
            config: Full project config dict. If None, loads from config.yaml.
        """
        if config is None:
            try:
                config = load_config()
            except FileNotFoundError:
                config = {}
        
        sev_cfg = config.get("severity", {})
        
        # Bounding-box area thresholds
        self.area_high_threshold = sev_cfg.get("area_high_threshold", 0.05)
        self.area_low_threshold = sev_cfg.get("area_low_threshold", 0.01)
        
        # Confidence thresholds
        self.conf_high_threshold = sev_cfg.get("conf_high_threshold", 0.70)
        self.conf_low_threshold = sev_cfg.get("conf_low_threshold", 0.40)
        
        # Score cutoffs
        self.high_score = sev_cfg.get("high_score", 1.5)
        self.medium_score = sev_cfg.get("medium_score", 0.5)
        
        logger.debug(f"SeverityCalculator initialized with thresholds: "
                     f"area_high={self.area_high_threshold}, "
                     f"conf_high={self.conf_high_threshold}")
    
    def compute_area_score(self, area_ratio: float) -> float:
        """
        Compute area-based severity score (0, 0.5, or 1.0).
        
        Args:
            area_ratio: Bounding-box area / image area
        
        Returns:
            0.0 (small), 0.5 (medium), or 1.0 (large)
        """
        if area_ratio >= self.area_high_threshold:
            return 1.0
        elif area_ratio < self.area_low_threshold:
            return 0.0
        else:
            return 0.5
    
    def compute_confidence_score(self, confidence: float) -> float:
        """
        Compute confidence-based severity score.
        
        Args:
            confidence: YOLO detection confidence (0.0 to 1.0)
        
        Returns:
            0.0 (low), 0.5 (medium), or 1.0 (high)
        """
        if confidence >= self.conf_high_threshold:
            return 1.0
        elif confidence < self.conf_low_threshold:
            return 0.0
        else:
            return 0.5
    
    def calculate(
        self,
        area_ratio: float,
        confidence: float
    ) -> str:
        """
        Calculate the final severity label for a detection.
        
        Project-defined method — NOT from RDD2022.
        
        Algorithm:
            area_score = score based on bounding-box area ratio
            conf_score = score based on confidence
            total = area_score + conf_score
            if total >= high_score  => High
            if total >= medium_score => Medium
            else                     => Low
        
        Args:
            area_ratio: Bounding-box area / image area (0.0 to 1.0)
            confidence: Detection confidence (0.0 to 1.0)
        
        Returns:
            'Low', 'Medium', or 'High'
        """
        area_score = self.compute_area_score(area_ratio)
        conf_score = self.compute_confidence_score(confidence)
        total_score = area_score + conf_score
        
        if total_score >= self.high_score:
            severity = "High"
        elif total_score >= self.medium_score:
            severity = "Medium"
        else:
            severity = "Low"
        
        logger.debug(f"Severity: area_ratio={area_ratio:.4f} "
                     f"area_score={area_score} conf={confidence:.4f} "
                     f"conf_score={conf_score} total={total_score} => {severity}")
        
        return severity
    
    def calculate_from_bbox(
        self,
        x1: float, y1: float, x2: float, y2: float,
        img_w: float, img_h: float,
        confidence: float
    ) -> Tuple[str, float]:
        """
        Calculate severity from raw bounding-box coordinates.
        
        Args:
            x1, y1, x2, y2: Bounding box pixel coordinates
            img_w, img_h: Image dimensions in pixels
            confidence: Detection confidence
        
        Returns:
            Tuple of (severity_label, area_ratio)
        """
        if img_w > 0 and img_h > 0:
            box_area = max(0, x2 - x1) * max(0, y2 - y1)
            area_ratio = box_area / (img_w * img_h)
        else:
            area_ratio = 0.0
        
        severity = self.calculate(area_ratio, confidence)
        return severity, area_ratio


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

_default_calculator = None


def get_severity(area_ratio: float, confidence: float, config: dict = None) -> str:
    """
    Get severity label using the default SeverityCalculator.
    
    Args:
        area_ratio: Bounding-box area / image area
        confidence: Detection confidence
        config: Optional config dict
    
    Returns:
        'Low', 'Medium', or 'High'
    """
    global _default_calculator
    if _default_calculator is None:
        _default_calculator = SeverityCalculator(config)
    return _default_calculator.calculate(area_ratio, confidence)


SEVERITY_LEVELS = ["Low", "Medium", "High"]
SEVERITY_COLORS = {
    "Low": "#2ecc71",    # Green
    "Medium": "#f39c12", # Orange
    "High": "#e74c3c"    # Red
}

# Backward-compatible ML helper. Existing inference modules call the
# arguments in (confidence, area_ratio) order, while get_severity() uses
# (area_ratio, confidence).
def calculate_severity(confidence: float, area_ratio: float, config: dict = None) -> str:
    return get_severity(area_ratio, confidence, config)
