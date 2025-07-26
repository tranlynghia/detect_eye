import cv2
import mediapipe as mp
import numpy as np
import math
from typing import Tuple, Optional, List, Dict
import time

class ImprovedEyeTracker:
    def __init__(self):
        """Initialize improved eye tracker with MediaPipe"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Face mesh configuration with iris landmarks
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Detailed eye landmark indices
        self.LEFT_EYE_LANDMARKS = [
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
        ]
        self.RIGHT_EYE_LANDMARKS = [
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
        ]
        
        # Iris landmarks (MediaPipe provides these)
        self.LEFT_IRIS_LANDMARKS = [474, 475, 476, 477]
        self.RIGHT_IRIS_LANDMARKS = [469, 470, 471, 472]
        
        # Eye corner landmarks for better gaze estimation
        self.LEFT_EYE_CORNERS = [33, 133]  # inner, outer corner
        self.RIGHT_EYE_CORNERS = [362, 263]  # inner, outer corner
        
        # Calibration data
        self.calibration_data = []
        self.is_calibrated = False
        self.gaze_mapping_matrix = None
        
        # Smoothing filter
        self.gaze_history = []
        self.history_length = 5
        
        # Screen parameters
        self.screen_width = 1920
        self.screen_height = 1080
        
        # Current gaze data
        self.current_gaze = {
            'left_eye_center': None,
            'right_eye_center': None,
            'left_iris_center': None,
            'right_iris_center': None,
            'left_pupil_ratio': (0.5, 0.5),
            'right_pupil_ratio': (0.5, 0.5),
            'gaze_direction': None,
            'screen_gaze_point': None,
            'confidence': 0.0
        }
    
    def extract_eye_region(self, frame: np.ndarray, landmarks: List, 
                          eye_indices: List) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """Extract eye region with better precision"""
        h, w = frame.shape[:2]
        
        # Get eye landmark coordinates
        eye_points = []
        for idx in eye_indices:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            eye_points.append([x, y])
        
        eye_points = np.array(eye_points)
        
        # Calculate bounding box with margin
        x_min, y_min = np.min(eye_points, axis=0)
        x_max, y_max = np.max(eye_points, axis=0)
        
        # Add margin
        margin = 10
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w, x_max + margin)
        y_max = min(h, y_max + margin)
        
        # Extract eye region
        eye_region = frame[y_min:y_max, x_min:x_max]
        
        # Calculate eye center
        eye_center = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
        
        return eye_region, (x_min, y_min), eye_center
    
    def get_iris_center(self, landmarks: List, iris_indices: List, 
                       frame_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Get precise iris center from MediaPipe landmarks"""
        h, w = frame_shape[:2]
        
        iris_points = []
        for idx in iris_indices:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            iris_points.append([x, y])
        
        iris_points = np.array(iris_points)
        iris_center = np.mean(iris_points, axis=0).astype(int)
        
        return tuple(iris_center)
    
    def calculate_pupil_ratio(self, iris_center: Tuple[int, int], 
                            eye_corners: List[Tuple[int, int]]) -> Tuple[float, float]:
        """Calculate pupil position ratio within eye"""
        if len(eye_corners) < 2:
            return (0.5, 0.5)
        
        inner_corner, outer_corner = eye_corners
        
        # Calculate eye width and height
        eye_width = abs(outer_corner[0] - inner_corner[0])
        eye_height = abs(outer_corner[1] - inner_corner[1]) + 20  # Add some height
        
        if eye_width == 0 or eye_height == 0:
            return (0.5, 0.5)
        
        # Calculate relative position
        rel_x = (iris_center[0] - inner_corner[0]) / eye_width
        rel_y = (iris_center[1] - min(inner_corner[1], outer_corner[1]) + 10) / eye_height
        
        # Clamp values
        rel_x = max(0.0, min(1.0, rel_x))
        rel_y = max(0.0, min(1.0, rel_y))
        
        return (rel_x, rel_y)
    
    def estimate_gaze_direction(self, left_pupil_ratio: Tuple[float, float], 
                              right_pupil_ratio: Tuple[float, float]) -> Tuple[float, float]:
        """Estimate gaze direction from pupil ratios"""
        # Average both eyes for more stable gaze estimation
        avg_x = (left_pupil_ratio[0] + right_pupil_ratio[0]) / 2.0
        avg_y = (left_pupil_ratio[1] + right_pupil_ratio[1]) / 2.0
        
        # Convert to normalized gaze direction (-1 to 1)
        gaze_x = (avg_x - 0.5) * 2.0
        gaze_y = (avg_y - 0.5) * 2.0
        
        return (gaze_x, gaze_y)
    
    def smooth_gaze(self, gaze_direction: Tuple[float, float]) -> Tuple[float, float]:
        """Apply smoothing filter to gaze direction"""
        self.gaze_history.append(gaze_direction)
        
        if len(self.gaze_history) > self.history_length:
            self.gaze_history.pop(0)
        
        # Calculate weighted average (more weight to recent samples)
        weights = np.linspace(0.5, 1.0, len(self.gaze_history))
        weights = weights / np.sum(weights)
        
        smooth_x = np.average([g[0] for g in self.gaze_history], weights=weights)
        smooth_y = np.average([g[1] for g in self.gaze_history], weights=weights)
        
        return (smooth_x, smooth_y)
    
    def gaze_to_screen_point(self, gaze_direction: Tuple[float, float]) -> Tuple[int, int]:
        """Convert gaze direction to screen coordinates"""
        if self.is_calibrated and self.gaze_mapping_matrix is not None:
            # Use calibrated mapping
            gx, gy = gaze_direction
            
            # Apply polynomial transformation
            screen_x = (self.gaze_mapping_matrix[0][0] * gx + 
                       self.gaze_mapping_matrix[0][1] * gy + 
                       self.gaze_mapping_matrix[0][2] * gx * gy + 
                       self.gaze_mapping_matrix[0][3] * gx**2 + 
                       self.gaze_mapping_matrix[0][4] * gy**2 + 
                       self.gaze_mapping_matrix[0][5])
            
            screen_y = (self.gaze_mapping_matrix[1][0] * gx + 
                       self.gaze_mapping_matrix[1][1] * gy + 
                       self.gaze_mapping_matrix[1][2] * gx * gy + 
                       self.gaze_mapping_matrix[1][3] * gx**2 + 
                       self.gaze_mapping_matrix[1][4] * gy**2 + 
                       self.gaze_mapping_matrix[1][5])
        else:
            # Simple linear mapping
            screen_x = (gaze_direction[0] + 1) * self.screen_width / 2
            screen_y = (gaze_direction[1] + 1) * self.screen_height / 2
        
        # Clamp to screen bounds
        screen_x = max(0, min(self.screen_width, int(screen_x)))
        screen_y = max(0, min(self.screen_height, int(screen_y)))
        
        return (screen_x, screen_y)
    
    def calculate_confidence(self, left_iris: Optional[Tuple[int, int]], 
                           right_iris: Optional[Tuple[int, int]]) -> float:
        """Calculate confidence score for gaze estimation"""
        confidence = 0.0
        
        if left_iris is not None:
            confidence += 0.5
        if right_iris is not None:
            confidence += 0.5
        
        # Reduce confidence if gaze history is short
        if len(self.gaze_history) < self.history_length:
            confidence *= len(self.gaze_history) / self.history_length
        
        return confidence
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process frame for improved eye tracking"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        annotated_frame = frame.copy()
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = face_landmarks.landmark
            h, w = frame.shape[:2]
            
            # Get eye centers
            left_eye_region, left_eye_offset, left_eye_center = self.extract_eye_region(
                frame, landmarks, self.LEFT_EYE_LANDMARKS
            )
            right_eye_region, right_eye_offset, right_eye_center = self.extract_eye_region(
                frame, landmarks, self.RIGHT_EYE_LANDMARKS
            )
            
            # Get iris centers
            left_iris_center = self.get_iris_center(landmarks, self.LEFT_IRIS_LANDMARKS, (h, w))
            right_iris_center = self.get_iris_center(landmarks, self.RIGHT_IRIS_LANDMARKS, (h, w))
            
            # Get eye corners for pupil ratio calculation
            left_corners = [(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) 
                           for idx in self.LEFT_EYE_CORNERS]
            right_corners = [(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) 
                            for idx in self.RIGHT_EYE_CORNERS]
            
            # Calculate pupil ratios
            left_pupil_ratio = self.calculate_pupil_ratio(left_iris_center, left_corners)
            right_pupil_ratio = self.calculate_pupil_ratio(right_iris_center, right_corners)
            
            # Estimate gaze direction
            raw_gaze_direction = self.estimate_gaze_direction(left_pupil_ratio, right_pupil_ratio)
            smooth_gaze_direction = self.smooth_gaze(raw_gaze_direction)
            
            # Convert to screen coordinates
            screen_gaze_point = self.gaze_to_screen_point(smooth_gaze_direction)
            
            # Calculate confidence
            confidence = self.calculate_confidence(left_iris_center, right_iris_center)
            
            # Update current gaze data
            self.current_gaze.update({
                'left_eye_center': left_eye_center,
                'right_eye_center': right_eye_center,
                'left_iris_center': left_iris_center,
                'right_iris_center': right_iris_center,
                'left_pupil_ratio': left_pupil_ratio,
                'right_pupil_ratio': right_pupil_ratio,
                'gaze_direction': smooth_gaze_direction,
                'screen_gaze_point': screen_gaze_point,
                'confidence': confidence
            })
            
            # Draw annotations
            annotated_frame = self.draw_detailed_annotations(annotated_frame)
        
        return annotated_frame, self.current_gaze
    
    def draw_detailed_annotations(self, frame: np.ndarray) -> np.ndarray:
        """Draw detailed eye tracking annotations"""
        # Draw eye centers
        if self.current_gaze['left_eye_center']:
            cv2.circle(frame, self.current_gaze['left_eye_center'], 8, (0, 255, 0), 2)
            cv2.putText(frame, 'L', 
                       (self.current_gaze['left_eye_center'][0] - 15, 
                        self.current_gaze['left_eye_center'][1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if self.current_gaze['right_eye_center']:
            cv2.circle(frame, self.current_gaze['right_eye_center'], 8, (0, 255, 0), 2)
            cv2.putText(frame, 'R', 
                       (self.current_gaze['right_eye_center'][0] + 10, 
                        self.current_gaze['right_eye_center'][1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw iris centers (pupils)
        if self.current_gaze['left_iris_center']:
            cv2.circle(frame, self.current_gaze['left_iris_center'], 4, (255, 0, 0), -1)
        
        if self.current_gaze['right_iris_center']:
            cv2.circle(frame, self.current_gaze['right_iris_center'], 4, (255, 0, 0), -1)
        
        # Draw gaze direction vector
        if (self.current_gaze['left_eye_center'] and self.current_gaze['right_eye_center'] 
            and self.current_gaze['gaze_direction']):
            
            # Calculate center point between eyes
            center_x = (self.current_gaze['left_eye_center'][0] + 
                       self.current_gaze['right_eye_center'][0]) // 2
            center_y = (self.current_gaze['left_eye_center'][1] + 
                       self.current_gaze['right_eye_center'][1]) // 2
            
            gaze_x, gaze_y = self.current_gaze['gaze_direction']
            
            # Draw gaze vector
            vector_length = 150
            end_x = int(center_x + gaze_x * vector_length)
            end_y = int(center_y + gaze_y * vector_length)
            
            cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), 
                           (0, 0, 255), 3, tipLength=0.3)
        
        # Draw pupil ratios as text
        info_y = 30
        if self.current_gaze['left_pupil_ratio']:
            left_ratio = self.current_gaze['left_pupil_ratio']
            cv2.putText(frame, f"Left Pupil: ({left_ratio[0]:.2f}, {left_ratio[1]:.2f})", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 25
        
        if self.current_gaze['right_pupil_ratio']:
            right_ratio = self.current_gaze['right_pupil_ratio']
            cv2.putText(frame, f"Right Pupil: ({right_ratio[0]:.2f}, {right_ratio[1]:.2f})", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 25
        
        if self.current_gaze['gaze_direction']:
            gaze_dir = self.current_gaze['gaze_direction']
            cv2.putText(frame, f"Gaze: ({gaze_dir[0]:.2f}, {gaze_dir[1]:.2f})", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 25
        
        if self.current_gaze['screen_gaze_point']:
            screen_point = self.current_gaze['screen_gaze_point']
            cv2.putText(frame, f"Screen: ({screen_point[0]}, {screen_point[1]})", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 25
        
        # Draw confidence
        confidence = self.current_gaze['confidence']
        color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.4 else (0, 0, 255)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def is_looking_at_point(self, target_point: Tuple[int, int], 
                           tolerance: int = 100) -> Tuple[bool, float]:
        """Check if user is looking at a specific point on screen"""
        if not self.current_gaze['screen_gaze_point']:
            return False, 0.0
        
        gaze_point = self.current_gaze['screen_gaze_point']
        
        # Calculate distance
        distance = math.sqrt(
            (gaze_point[0] - target_point[0]) ** 2 + 
            (gaze_point[1] - target_point[1]) ** 2
        )
        
        # Check if within tolerance
        is_looking = distance <= tolerance
        
        # Calculate accuracy (inverse of distance, normalized)
        max_distance = math.sqrt(self.screen_width**2 + self.screen_height**2)
        accuracy = max(0, 1 - (distance / max_distance))
        
        return is_looking, accuracy

def main():
    """Test the improved eye tracker"""
    eye_tracker = ImprovedEyeTracker()
    cap = cv2.VideoCapture(0)
    
    # Test target points on screen
    target_points = [
        (480, 270),   # Center
        (200, 150),   # Top-left
        (760, 150),   # Top-right
        (200, 390),   # Bottom-left
        (760, 390),   # Bottom-right
    ]
    current_target = 0
    
    print("Improved Eye Tracking Test")
    print("Controls:")
    print("  'q' - Quit")
    print("  'n' - Next target point")
    print("  'c' - Start calibration")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        annotated_frame, gaze_data = eye_tracker.process_frame(frame)
        
        # Draw current target point
        target_point = target_points[current_target]
        # Scale target point to frame size
        frame_h, frame_w = frame.shape[:2]
        scaled_target = (
            int(target_point[0] * frame_w / eye_tracker.screen_width),
            int(target_point[1] * frame_h / eye_tracker.screen_height)
        )
        
        cv2.circle(annotated_frame, scaled_target, 20, (0, 255, 255), 3)
        cv2.putText(annotated_frame, f"Target {current_target + 1}", 
                   (scaled_target[0] - 30, scaled_target[1] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Check if looking at target
        is_looking, accuracy = eye_tracker.is_looking_at_point(target_point, tolerance=150)
        
        if is_looking:
            cv2.putText(annotated_frame, "LOOKING AT TARGET!", 
                       (10, frame_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.putText(annotated_frame, f"Accuracy: {accuracy:.2f}", 
                       (10, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Improved Eye Tracking', annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            current_target = (current_target + 1) % len(target_points)
            print(f"Switched to target {current_target + 1}")
        elif key == ord('c'):
            print("Calibration mode not implemented in this demo")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
