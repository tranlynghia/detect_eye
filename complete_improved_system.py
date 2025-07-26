#!/usr/bin/env python3
"""
Complete Improved Eye Tracking System
Enhanced accuracy for pupil detection and screen point detection
"""

import cv2
import numpy as np
import argparse
import time
from pathlib import Path
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from improved_eye_tracker import ImprovedEyeTracker
from screen_point_detector import ScreenPointDetector
from calibration_improved import ImprovedCalibrationSystem

class CompleteImprovedSystem:
    def __init__(self):
        """Initialize the complete improved system"""
        self.eye_tracker = ImprovedEyeTracker()
        self.point_detector = ScreenPointDetector(self.eye_tracker)
        self.calibration_system = ImprovedCalibrationSystem(self.eye_tracker)
        
        # System state
        self.is_running = False
        self.show_debug_info = True
        self.show_heatmap = False
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Test targets for demonstration
        self.demo_targets = [
            (640, 360, "Center", (0, 255, 255)),
            (320, 180, "Top-Left", (255, 0, 255)),
            (960, 180, "Top-Right", (255, 255, 0)),
            (320, 540, "Bottom-Left", (0, 255, 0)),
            (960, 540, "Bottom-Right", (255, 0, 0))
        ]
        self.current_target_idx = 0
    
    def run_system(self, camera_id: int = 0):
        """Run the complete improved system"""
        print("🚀 Starting Complete Improved Eye Tracking System")
        print("=" * 60)
        
        # Try to load existing calibration
        if self.calibration_system.load_calibration():
            print("✅ Calibration loaded successfully")
        else:
            print("⚠️ No calibration found. System will use default mapping.")
            choice = input("Run calibration now? (y/n): ").lower()
            if choice == 'y':
                self.calibration_system.start_calibration()
                if self.calibration_system.calibration_complete:
                    self.calibration_system.save_calibration()
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("❌ Could not open camera")
            return
        
        self.is_running = True
        
        print("\n🎮 System Controls:")
        print("  'q' - Quit")
        print("  'c' - Start calibration")
        print("  'n' - Next demo target")
        print("  'd' - Toggle debug info")
        print("  'h' - Toggle attention heatmap")
        print("  'r' - Reset tracking")
        print("  's' - Save screenshot")
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Display result
            cv2.imshow('Complete Improved Eye Tracking System', processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.start_calibration()
            elif key == ord('n'):
                self.next_demo_target()
            elif key == ord('d'):
                self.show_debug_info = not self.show_debug_info
                print(f"Debug info: {'ON' if self.show_debug_info else 'OFF'}")
            elif key == ord('h'):
                self.show_heatmap = not self.show_heatmap
                print(f"Heatmap: {'ON' if self.show_heatmap else 'OFF'}")
            elif key == ord('r'):
                self.reset_system()
            elif key == ord('s'):
                self.save_screenshot(processed_frame)
        
        cap.release()
        cv2.destroyAllWindows()
        print("🛑 System stopped")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame through all system components"""
        # Update FPS counter
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
        
        # Process eye tracking
        annotated_frame, gaze_data = self.eye_tracker.process_frame(frame)
        
        # Current demo target
        target_x, target_y, target_name, target_color = self.demo_targets[self.current_target_idx]
        
        # Check if looking at current target
        is_looking, accuracy, analysis = self.point_detector.is_looking_at_specific_point(
            (target_x, target_y), tolerance=120
        )
        
        # Draw demo target
        frame_h, frame_w = frame.shape[:2]
        scaled_x = int(target_x * frame_w / self.eye_tracker.screen_width)
        scaled_y = int(target_y * frame_h / self.eye_tracker.screen_height)
        
        # Target appearance changes based on detection
        if is_looking:
            cv2.circle(annotated_frame, (scaled_x, scaled_y), 35, (0, 255, 0), 5)
            cv2.circle(annotated_frame, (scaled_x, scaled_y), 15, (0, 255, 0), -1)
        else:
            cv2.circle(annotated_frame, (scaled_x, scaled_y), 30, target_color, 3)
            cv2.circle(annotated_frame, (scaled_x, scaled_y), 8, target_color, -1)
        
        # Target label
        cv2.putText(annotated_frame, target_name, 
                   (scaled_x - 40, scaled_y - 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, target_color, 2)
        
        # Draw detection overlay
        annotated_frame = self.point_detector.draw_detection_overlay(annotated_frame)
        
        # Show attention heatmap if enabled
        if self.show_heatmap:
            heatmap = self.point_detector.create_attention_heatmap(frame.shape[:2])
            if np.max(heatmap) > 0:
                heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
                annotated_frame = cv2.addWeighted(annotated_frame, 0.7, heatmap_colored, 0.3, 0)
        
        # Draw system status and debug info
        if self.show_debug_info:
            annotated_frame = self.draw_system_status(annotated_frame, gaze_data, analysis, is_looking)
        
        return annotated_frame
    
    def draw_system_status(self, frame: np.ndarray, gaze_data: dict, 
                          analysis: dict, is_looking: bool) -> np.ndarray:
        """Draw comprehensive system status"""
        h, w = frame.shape[:2]
        
        # Status panel background
        panel_height = 200
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - panel_height - 10), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # System status
        y_offset = h - panel_height + 20
        status_texts = [
            f"FPS: {self.current_fps:.1f}",
            f"Calibrated: {'Yes' if self.eye_tracker.is_calibrated else 'No'}",
            f"Confidence: {gaze_data.get('confidence', 0):.2f}",
            f"Looking at target: {'YES' if is_looking else 'NO'}",
            f"Accuracy: {analysis.get('accuracy_score', 0):.2f}",
            f"Distance: {analysis.get('distance_to_target', 0):.1f}px",
            f"Fixating: {'Yes' if analysis.get('fixation_detected', False) else 'No'}",
            f"Gaze Stability: {analysis.get('gaze_stability', 0):.2f}"
        ]
        
        # Draw status texts
        for i, text in enumerate(status_texts):
            color = (0, 255, 0) if 'YES' in text else (255, 255, 255)
            cv2.putText(frame, text, (20, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw gaze direction indicator
        if gaze_data.get('gaze_direction'):
            gaze_x, gaze_y = gaze_data['gaze_direction']
            
            # Gaze direction compass
            compass_center = (w - 100, h - 100)
            compass_radius = 40
            
            cv2.circle(frame, compass_center, compass_radius, (100, 100, 100), 2)
            cv2.circle(frame, compass_center, 3, (255, 255, 255), -1)
            
            # Gaze direction arrow
            arrow_end = (
                int(compass_center[0] + gaze_x * compass_radius * 0.8),
                int(compass_center[1] + gaze_y * compass_radius * 0.8)
            )
            cv2.arrowedLine(frame, compass_center, arrow_end, (0, 255, 255), 3)
            
            cv2.putText(frame, "Gaze", (compass_center[0] - 20, compass_center[1] + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def start_calibration(self):
        """Start calibration process"""
        print("🎯 Starting calibration...")
        self.calibration_system.start_calibration()
        if self.calibration_system.calibration_complete:
            self.calibration_system.save_calibration()
            print("✅ Calibration completed and saved")
    
    def next_demo_target(self):
        """Switch to next demo target"""
        self.current_target_idx = (self.current_target_idx + 1) % len(self.demo_targets)
        target_name = self.demo_targets[self.current_target_idx][2]
        print(f"🎯 Switched to target: {target_name}")
    
    def reset_system(self):
        """Reset system tracking data"""
        self.eye_tracker.gaze_history = []
        self.point_detector.fixation_history = []
        self.point_detector.reset_fixation()
        print("🔄 System reset completed")
    
    def save_screenshot(self, frame: np.ndarray):
        """Save current frame as screenshot"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"eye_tracking_screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Complete Improved Eye Tracking System')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--calibrate', action='store_true', help='Start with calibration')
    parser.add_argument('--load-calibration', type=str, help='Load calibration file')
    
    args = parser.parse_args()
    
    try:
        system = CompleteImprovedSystem()
        
        # Load specific calibration if provided
        if args.load_calibration:
            system.calibration_system.load_calibration(args.load_calibration)
        
        # Start with calibration if requested
        if args.calibrate:
            system.calibration_system.start_calibration()
            if system.calibration_system.calibration_complete:
                system.calibration_system.save_calibration()
        
        # Run the main system
        system.run_system(args.camera)
        
    except KeyboardInterrupt:
        print("\n⚠️ System interrupted by user")
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
