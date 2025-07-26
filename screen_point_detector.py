import cv2
import numpy as np
import math
import time
from typing import Tuple, List, Dict, Optional
from improved_eye_tracker import ImprovedEyeTracker

class ScreenPointDetector:
    def __init__(self, eye_tracker: ImprovedEyeTracker):
        """Initialize screen point detector"""
        self.eye_tracker = eye_tracker
        
        # Detection parameters
        self.fixation_threshold = 100  # pixels
        self.fixation_duration_threshold = 1.0  # seconds
        self.confidence_threshold = 0.5
        
        # Tracking data
        self.current_fixation = None
        self.fixation_start_time = None
        self.fixation_history = []
        
        # Screen regions for detection
        self.screen_regions = {
            'top_left': (0, 0, 640, 360),
            'top_right': (640, 0, 1280, 360),
            'center': (480, 270, 1440, 810),
            'bottom_left': (0, 720, 640, 1080),
            'bottom_right': (640, 720, 1280, 1080)
        }
        
        # Detection results
        self.detection_results = {
            'is_fixating': False,
            'fixation_point': None,
            'fixation_duration': 0.0,
            'fixation_region': None,
            'gaze_stability': 0.0,
            'detection_confidence': 0.0
        }
    
    def detect_fixation(self, gaze_point: Tuple[int, int], 
                       confidence: float) -> Dict:
        """Detect if user is fixating on a point"""
        current_time = time.time()
        
        if confidence < self.confidence_threshold:
            self.reset_fixation()
            return self.detection_results
        
        if self.current_fixation is None:
            # Start new fixation
            self.current_fixation = gaze_point
            self.fixation_start_time = current_time
            self.detection_results['is_fixating'] = False
        else:
            # Check if still fixating on same point
            distance = math.sqrt(
                (gaze_point[0] - self.current_fixation[0]) ** 2 + 
                (gaze_point[1] - self.current_fixation[1]) ** 2
            )
            
            if distance <= self.fixation_threshold:
                # Continue fixation
                fixation_duration = current_time - self.fixation_start_time
                
                if fixation_duration >= self.fixation_duration_threshold:
                    # Valid fixation detected
                    self.detection_results.update({
                        'is_fixating': True,
                        'fixation_point': self.current_fixation,
                        'fixation_duration': fixation_duration,
                        'fixation_region': self.get_screen_region(self.current_fixation),
                        'gaze_stability': self.calculate_gaze_stability(),
                        'detection_confidence': confidence
                    })
                else:
                    self.detection_results['is_fixating'] = False
            else:
                # Fixation broken, start new one
                self.current_fixation = gaze_point
                self.fixation_start_time = current_time
                self.detection_results['is_fixating'] = False
        
        return self.detection_results
    
    def reset_fixation(self):
        """Reset fixation tracking"""
        self.current_fixation = None
        self.fixation_start_time = None
        self.detection_results.update({
            'is_fixating': False,
            'fixation_point': None,
            'fixation_duration': 0.0,
            'fixation_region': None,
            'gaze_stability': 0.0,
            'detection_confidence': 0.0
        })
    
    def get_screen_region(self, point: Tuple[int, int]) -> Optional[str]:
        """Determine which screen region the point is in"""
        x, y = point
        
        for region_name, (x1, y1, x2, y2) in self.screen_regions.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return region_name
        
        return None
    
    def calculate_gaze_stability(self) -> float:
        """Calculate gaze stability score"""
        if len(self.eye_tracker.gaze_history) < 3:
            return 0.0
        
        # Calculate variance in recent gaze points
        recent_gazes = self.eye_tracker.gaze_history[-5:]
        
        x_coords = [g[0] for g in recent_gazes]
        y_coords = [g[1] for g in recent_gazes]
        
        x_var = np.var(x_coords)
        y_var = np.var(y_coords)
        
        # Convert to stability score (lower variance = higher stability)
        stability = 1.0 / (1.0 + x_var + y_var)
        
        return min(1.0, stability)
    
    def is_looking_at_specific_point(self, target_point: Tuple[int, int], 
                                   tolerance: int = 80) -> Tuple[bool, float, Dict]:
        """Check if user is looking at a specific point with detailed analysis"""
        gaze_data = self.eye_tracker.current_gaze
        
        if not gaze_data['screen_gaze_point']:
            return False, 0.0, {}
        
        gaze_point = gaze_data['screen_gaze_point']
        confidence = gaze_data['confidence']
        
        # Calculate distance to target
        distance = math.sqrt(
            (gaze_point[0] - target_point[0]) ** 2 + 
            (gaze_point[1] - target_point[1]) ** 2
        )
        
        # Check if within tolerance
        is_looking = distance <= tolerance and confidence > self.confidence_threshold
        
        # Calculate accuracy score
        accuracy = max(0, 1 - (distance / tolerance)) if tolerance > 0 else 0
        
        # Detect fixation on this point
        fixation_data = self.detect_fixation(gaze_point, confidence)
        
        # Additional analysis
        analysis = {
            'distance_to_target': distance,
            'within_tolerance': is_looking,
            'accuracy_score': accuracy,
            'gaze_confidence': confidence,
            'fixation_detected': fixation_data['is_fixating'],
            'fixation_duration': fixation_data['fixation_duration'],
            'gaze_stability': fixation_data['gaze_stability']
        }
        
        return is_looking, accuracy, analysis
    
    def create_attention_heatmap(self, frame_shape: Tuple[int, int], 
                               history_length: int = 50) -> np.ndarray:
        """Create attention heatmap from gaze history"""
        h, w = frame_shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        if len(self.fixation_history) == 0:
            return heatmap
        
        # Get recent fixation points
        recent_fixations = self.fixation_history[-history_length:]
        
        for fixation_point, duration in recent_fixations:
            # Scale to frame size
            fx = int(fixation_point[0] * w / self.eye_tracker.screen_width)
            fy = int(fixation_point[1] * h / self.eye_tracker.screen_height)
            
            # Add gaussian blob
            sigma = 30
            for y in range(max(0, fy - sigma * 2), min(h, fy + sigma * 2)):
                for x in range(max(0, fx - sigma * 2), min(w, fx + sigma * 2)):
                    distance = math.sqrt((x - fx) ** 2 + (y - fy) ** 2)
                    intensity = math.exp(-(distance ** 2) / (2 * sigma ** 2))
                    heatmap[y, x] += intensity * duration
        
        # Normalize
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        return heatmap
    
    def draw_detection_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw detection overlay on frame"""
        overlay = frame.copy()
        
        # Draw current fixation
        if self.detection_results['is_fixating'] and self.detection_results['fixation_point']:
            fixation_point = self.detection_results['fixation_point']
            
            # Scale to frame coordinates
            frame_h, frame_w = frame.shape[:2]
            fx = int(fixation_point[0] * frame_w / self.eye_tracker.screen_width)
            fy = int(fixation_point[1] * frame_h / self.eye_tracker.screen_height)
            
            # Draw fixation circle
            radius = int(20 + self.detection_results['fixation_duration'] * 10)
            cv2.circle(overlay, (fx, fy), radius, (0, 255, 0), 3)
            cv2.circle(overlay, (fx, fy), 5, (0, 255, 0), -1)
            
            # Draw fixation info
            cv2.putText(overlay, f"Fixation: {self.detection_results['fixation_duration']:.1f}s", 
                       (fx + 25, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            region = self.detection_results['fixation_region']
            if region:
                cv2.putText(overlay, f"Region: {region}", 
                           (fx + 25, fy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw screen regions
        frame_h, frame_w = frame.shape[:2]
        for region_name, (x1, y1, x2, y2) in self.screen_regions.items():
            # Scale to frame coordinates
            fx1 = int(x1 * frame_w / self.eye_tracker.screen_width)
            fy1 = int(y1 * frame_h / self.eye_tracker.screen_height)
            fx2 = int(x2 * frame_w / self.eye_tracker.screen_width)
            fy2 = int(y2 * frame_h / self.eye_tracker.screen_height)
            
            cv2.rectangle(overlay, (fx1, fy1), (fx2, fy2), (100, 100, 100), 1)
            cv2.putText(overlay, region_name, (fx1 + 5, fy1 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        return overlay

def main():
    """Test screen point detection"""
    eye_tracker = ImprovedEyeTracker()
    point_detector = ScreenPointDetector(eye_tracker)
    cap = cv2.VideoCapture(0)
    
    # Test targets
    test_targets = [
        (640, 360, "Center"),
        (320, 180, "Top-Left"),
        (960, 180, "Top-Right"),
        (320, 540, "Bottom-Left"),
        (960, 540, "Bottom-Right")
    ]
    current_target_idx = 0
    
    print("Screen Point Detection Test")
    print("Look at the yellow targets to test detection")
    print("Controls:")
    print("  'q' - Quit")
    print("  'n' - Next target")
    print("  'h' - Toggle heatmap")
    
    show_heatmap = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process eye tracking
        annotated_frame, gaze_data = eye_tracker.process_frame(frame)
        
        # Current target
        target_x, target_y, target_name = test_targets[current_target_idx]
        
        # Check if looking at target
        is_looking, accuracy, analysis = point_detector.is_looking_at_specific_point(
            (target_x, target_y), tolerance=100
        )
        
        # Draw target
        frame_h, frame_w = frame.shape[:2]
        scaled_x = int(target_x * frame_w / eye_tracker.screen_width)
        scaled_y = int(target_y * frame_h / eye_tracker.screen_height)
        
        color = (0, 255, 0) if is_looking else (0, 255, 255)
        cv2.circle(annotated_frame, (scaled_x, scaled_y), 25, color, 3)
        cv2.putText(annotated_frame, target_name, 
                   (scaled_x - 40, scaled_y - 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw detection overlay
        annotated_frame = point_detector.draw_detection_overlay(annotated_frame)
        
        # Show analysis info
        info_y = frame_h - 120
        analysis_texts = [
            f"Looking at target: {is_looking}",
            f"Accuracy: {accuracy:.2f}",
            f"Distance: {analysis.get('distance_to_target', 0):.1f}px",
            f"Fixating: {analysis.get('fixation_detected', False)}",
            f"Stability: {analysis.get('gaze_stability', 0):.2f}"
        ]
        
        for text in analysis_texts:
            cv2.putText(annotated_frame, text, (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated_frame, text, (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            info_y += 20
        
        # Show heatmap if enabled
        if show_heatmap:
            heatmap = point_detector.create_attention_heatmap(frame.shape[:2])
            heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            annotated_frame = cv2.addWeighted(annotated_frame, 0.7, heatmap_colored, 0.3, 0)
        
        cv2.imshow('Screen Point Detection', annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            current_target_idx = (current_target_idx + 1) % len(test_targets)
            print(f"Switched to target: {test_targets[current_target_idx][2]}")
        elif key == ord('h'):
            show_heatmap = not show_heatmap
            print(f"Heatmap: {'ON' if show_heatmap else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
