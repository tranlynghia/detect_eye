import cv2
import numpy as np
import time
import json
from typing import List, Tuple, Dict
from improved_eye_tracker import ImprovedEyeTracker

class ImprovedCalibrationSystem:
    def __init__(self, eye_tracker: ImprovedEyeTracker):
        """Initialize improved calibration system"""
        self.eye_tracker = eye_tracker
        
        # Calibration parameters
        self.calibration_points = []
        self.gaze_samples = []
        self.samples_per_point = 60  # 2 seconds at 30 FPS
        self.current_point_index = 0
        self.current_samples = []
        
        # Screen parameters
        self.screen_width = 1920
        self.screen_height = 1080
        
        # Generate calibration grid (5x3 for better accuracy)
        self.calibration_targets = self.generate_calibration_grid()
        
        # Calibration state
        self.is_calibrating = False
        self.calibration_complete = False
        self.calibration_accuracy = 0.0
        
    def generate_calibration_grid(self) -> List[Tuple[int, int]]:
        """Generate calibration target positions in a 5x3 grid"""
        targets = []
        
        # Grid dimensions
        cols, rows = 5, 3
        
        # Margins from screen edges
        margin_x = self.screen_width * 0.1
        margin_y = self.screen_height * 0.15
        
        for row in range(rows):
            for col in range(cols):
                x = int(margin_x + col * (self.screen_width - 2 * margin_x) / (cols - 1))
                y = int(margin_y + row * (self.screen_height - 2 * margin_y) / (rows - 1))
                targets.append((x, y))
        
        return targets
    
    def start_calibration(self):
        """Start the calibration process"""
        print("🎯 Starting improved calibration...")
        print(f"Will calibrate {len(self.calibration_targets)} points")
        
        self.is_calibrating = True
        self.current_point_index = 0
        self.calibration_points = []
        self.gaze_samples = []
        self.current_samples = []
        
        cap = cv2.VideoCapture(0)
        
        # Create fullscreen calibration window
        cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        start_time = time.time()
        point_start_time = time.time()
        
        while self.is_calibrating and self.current_point_index < len(self.calibration_targets):
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Process eye tracking
            _, gaze_data = self.eye_tracker.process_frame(frame)
            
            # Create calibration display
            cal_display = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            
            # Current target
            target_point = self.calibration_targets[self.current_point_index]
            
            # Animate target (pulsing circle)
            current_time = time.time()
            pulse = 0.5 + 0.5 * np.sin((current_time - point_start_time) * 4)
            radius = int(20 + pulse * 15)
            
            # Draw target
            cv2.circle(cal_display, target_point, radius, (0, 255, 255), -1)
            cv2.circle(cal_display, target_point, radius + 5, (255, 255, 255), 3)
            
            # Draw progress
            progress = len(self.current_samples) / self.samples_per_point
            progress_text = f"Point {self.current_point_index + 1}/{len(self.calibration_targets)}"
            cv2.putText(cal_display, progress_text, (50, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            # Progress bar
            bar_width = 400
            bar_height = 20
            bar_x, bar_y = 50, 120
            
            cv2.rectangle(cal_display, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            cv2.rectangle(cal_display, (bar_x, bar_y), 
                         (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
            
            # Instructions
            instruction = "Look at the yellow dot and keep your head still"
            cv2.putText(cal_display, instruction, (50, self.screen_height - 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            samples_text = f"Samples: {len(self.current_samples)}/{self.samples_per_point}"
            cv2.putText(cal_display, samples_text, (50, self.screen_height - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Calibration', cal_display)
            
            # Collect gaze samples
            if (gaze_data['gaze_direction'] is not None and 
                gaze_data['confidence'] > 0.5):
                
                self.current_samples.append(gaze_data['gaze_direction'])
                
                # Check if enough samples collected
                if len(self.current_samples) >= self.samples_per_point:
                    # Calculate average gaze for this point
                    avg_gaze_x = np.mean([s[0] for s in self.current_samples])
                    avg_gaze_y = np.mean([s[1] for s in self.current_samples])
                    
                    # Store calibration data
                    self.calibration_points.append(target_point)
                    self.gaze_samples.append((avg_gaze_x, avg_gaze_y))
                    
                    print(f"✅ Point {self.current_point_index + 1} calibrated: "
                          f"Target({target_point[0]}, {target_point[1]}) -> "
                          f"Gaze({avg_gaze_x:.3f}, {avg_gaze_y:.3f})")
                    
                    # Move to next point
                    self.current_point_index += 1
                    self.current_samples = []
                    point_start_time = time.time()
                    
                    # Brief pause between points
                    time.sleep(0.5)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):  # Reset current point
                self.current_samples = []
                point_start_time = time.time()
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Calculate calibration matrix
        if len(self.calibration_points) >= 9:  # Minimum points for good calibration
            self.calculate_calibration_matrix()
            self.validate_calibration()
            self.calibration_complete = True
            print(f"🎉 Calibration completed! Accuracy: {self.calibration_accuracy:.1f}px")
        else:
            print("❌ Calibration failed - insufficient data points")
        
        self.is_calibrating = False
    
    def calculate_calibration_matrix(self):
        """Calculate transformation matrix using polynomial regression"""
        if len(self.calibration_points) < 6:
            return
        
        # Convert to numpy arrays
        screen_points = np.array(self.calibration_points, dtype=np.float32)
        gaze_points = np.array(self.gaze_samples, dtype=np.float32)
        
        try:
            # Create feature matrix for 2nd order polynomial transformation
            # Features: [x, y, xy, x², y², 1]
            X = np.column_stack([
                gaze_points[:, 0],                      # x
                gaze_points[:, 1],                      # y
                gaze_points[:, 0] * gaze_points[:, 1],  # xy
                gaze_points[:, 0] ** 2,                 # x²
                gaze_points[:, 1] ** 2,                 # y²
                np.ones(len(gaze_points))               # constant term
            ])
            
            # Solve for transformation coefficients using least squares
            coeffs_x = np.linalg.lstsq(X, screen_points[:, 0], rcond=None)[0]
            coeffs_y = np.linalg.lstsq(X, screen_points[:, 1], rcond=None)[0]
            
            # Store calibration matrix
            self.eye_tracker.gaze_mapping_matrix = (coeffs_x, coeffs_y)
            self.eye_tracker.is_calibrated = True
            
            print("✅ Calibration matrix calculated successfully")
            
        except np.linalg.LinAlgError as e:
            print(f"❌ Failed to calculate calibration matrix: {e}")
    
    def validate_calibration(self):
        """Validate calibration accuracy"""
        if not self.eye_tracker.is_calibrated:
            return
        
        errors = []
        
        for i, (screen_point, gaze_point) in enumerate(zip(self.calibration_points, self.gaze_samples)):
            # Transform gaze to screen coordinates
            predicted_screen = self.eye_tracker.gaze_to_screen_point(gaze_point)
            
            # Calculate error
            error = np.sqrt((predicted_screen[0] - screen_point[0])**2 + 
                           (predicted_screen[1] - screen_point[1])**2)
            errors.append(error)
        
        self.calibration_accuracy = np.mean(errors)
        max_error = np.max(errors)
        std_error = np.std(errors)
        
        print(f"📊 Calibration Validation Results:")
        print(f"  Average error: {self.calibration_accuracy:.1f} pixels")
        print(f"  Maximum error: {max_error:.1f} pixels")
        print(f"  Standard deviation: {std_error:.1f} pixels")
        print(f"  RMS error: {np.sqrt(np.mean(np.array(errors)**2)):.1f} pixels")
        
        # Quality assessment
        if self.calibration_accuracy < 50:
            print("✅ Excellent calibration quality")
        elif self.calibration_accuracy < 100:
            print("⚠️ Good calibration quality")
        elif self.calibration_accuracy < 150:
            print("⚠️ Fair calibration quality - consider recalibrating")
        else:
            print("❌ Poor calibration quality - recalibration recommended")
    
    def save_calibration(self, filename: str = "improved_calibration.json"):
        """Save calibration data to file"""
        if not self.calibration_complete:
            print("❌ No calibration data to save")
            return False
        
        calibration_data = {
            'screen_width': self.screen_width,
            'screen_height': self.screen_height,
            'calibration_points': self.calibration_points,
            'gaze_samples': self.gaze_samples,
            'calibration_matrix': {
                'coeffs_x': self.eye_tracker.gaze_mapping_matrix[0].tolist(),
                'coeffs_y': self.eye_tracker.gaze_mapping_matrix[1].tolist()
            } if self.eye_tracker.gaze_mapping_matrix else None,
            'calibration_accuracy': self.calibration_accuracy,
            'timestamp': time.time(),
            'samples_per_point': self.samples_per_point
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            print(f"💾 Calibration saved to {filename}")
            return True
        except Exception as e:
            print(f"❌ Failed to save calibration: {e}")
            return False
    
    def load_calibration(self, filename: str = "improved_calibration.json"):
        """Load calibration data from file"""
        try:
            with open(filename, 'r') as f:
                calibration_data = json.load(f)
            
            self.screen_width = calibration_data['screen_width']
            self.screen_height = calibration_data['screen_height']
            self.calibration_points = calibration_data['calibration_points']
            self.gaze_samples = calibration_data['gaze_samples']
            self.calibration_accuracy = calibration_data.get('calibration_accuracy', 0.0)
            
            if calibration_data['calibration_matrix']:
                coeffs_x = np.array(calibration_data['calibration_matrix']['coeffs_x'])
                coeffs_y = np.array(calibration_data['calibration_matrix']['coeffs_y'])
                self.eye_tracker.gaze_mapping_matrix = (coeffs_x, coeffs_y)
                self.eye_tracker.is_calibrated = True
                self.calibration_complete = True
            
            print(f"📂 Calibration loaded from {filename}")
            print(f"   Accuracy: {self.calibration_accuracy:.1f} pixels")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load calibration: {e}")
            return False

def main():
    """Test improved calibration system"""
    eye_tracker = ImprovedEyeTracker()
    calibration_system = ImprovedCalibrationSystem(eye_tracker)
    
    print("🎯 Improved Calibration System")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Start new calibration")
        print("2. Load existing calibration")
        print("3. Test current calibration")
        print("4. Exit")
        
        choice = input("Choose option (1-4): ").strip()
        
        if choice == '1':
            calibration_system.start_calibration()
            if calibration_system.calibration_complete:
                calibration_system.save_calibration()
        
        elif choice == '2':
            calibration_system.load_calibration()
        
        elif choice == '3':
            if not eye_tracker.is_calibrated:
                print("❌ No calibration loaded. Please calibrate first.")
                continue
            
            # Test calibration
            from screen_point_detector import main as test_detection
            test_detection()
        
        elif choice == '4':
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
