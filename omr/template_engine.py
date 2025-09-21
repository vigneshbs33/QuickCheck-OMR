"""
Template-based OMR Engine for QuickCheck
Uses template matching and fixed grid positions for reliable detection
Developed by InteliCat Team
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path


class TemplateOMRProcessor:
    """
    Template-based OMR processor that uses fixed grid positions
    Optimized for the specific OMR format with known layout
    """
    
    def __init__(self):
        self.questions_per_column = 20
        self.options_per_question = 4
        self.total_questions = 100
        self.num_columns = 5
        
        # These will be set based on the actual image dimensions
        self.grid_bounds = None
        self.bubble_positions = None
        
    def process_omr(self, image: np.ndarray, answer_key: Dict[str, str], 
                   subjects: Dict[str, str], bubble_threshold: float = 0.4) -> Optional[Dict[str, Any]]:
        """
        Main processing function for OMR evaluation
        """
        try:
            print(f"Processing image of shape: {image.shape}")
            
            # Step 1: Detect and correct orientation
            corrected_image = self.correct_orientation(image)
            
            # Step 2: Detect the OMR grid
            grid_bounds = self.detect_omr_grid(corrected_image)
            if grid_bounds is None:
                print("Could not detect OMR grid")
                return None
            
            # Step 3: Build bubble positions based on grid
            bubble_positions = self.build_bubble_positions(corrected_image, grid_bounds)
            if len(bubble_positions) < 50:  # Need at least 50 questions worth
                print(f"Could not build bubble positions, only {len(bubble_positions)} found")
                return None
            
            # Step 4: Detect filled bubbles
            filled_bubbles = self.detect_filled_bubbles(corrected_image, bubble_positions, bubble_threshold)
            
            # Step 5: Evaluate answers
            results = self.evaluate_answers(filled_bubbles, answer_key, subjects)
            
            # Step 6: Create annotated image
            annotated_image = self.create_annotated_image(corrected_image, bubble_positions, filled_bubbles, results)
            results['annotated_image'] = annotated_image
            
            return results
            
        except Exception as e:
            print(f"Error in OMR processing: {str(e)}")
            return None
    
    def correct_orientation(self, image: np.ndarray) -> np.ndarray:
        """Correct image orientation using edge detection"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None and len(lines) > 0:
                # Calculate average angle
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angle = theta * 180 / np.pi
                    if angle > 90:
                        angle -= 180
                    angles.append(angle)
                
                if angles:
                    avg_angle = np.median(angles)
                    
                    # Rotate image if angle is significant
                    if abs(avg_angle) > 1:
                        h, w = image.shape[:2]
                        center = (w // 2, h // 2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
                        rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
                        return rotated
            
            return image
            
        except Exception as e:
            print(f"Error in orientation correction: {str(e)}")
            return image
    
    def detect_omr_grid(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect the main OMR grid area"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive threshold
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the largest rectangular area
            max_area = 0
            best_rect = None
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    # Check if it's roughly the right aspect ratio for OMR
                    aspect_ratio = w / h
                    if 0.5 < aspect_ratio < 2.0:  # Reasonable aspect ratio
                        max_area = area
                        best_rect = (x, y, w, h)
            
            if best_rect is not None:
                print(f"Detected grid area: {best_rect}")
                return best_rect
            
            return None
            
        except Exception as e:
            print(f"Error detecting OMR grid: {str(e)}")
            return None
    
    def build_bubble_positions(self, image: np.ndarray, grid_bounds: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """Build bubble positions based on detected grid"""
        try:
            x, y, w, h = grid_bounds
            
            # Calculate positions based on the known OMR format
            # 5 columns, 20 questions per column, 4 options per question
            
            bubble_positions = {}
            question_num = 1
            
            # Calculate column width
            col_width = w // 5
            
            # Calculate row height (approximate)
            row_height = h // 25  # 20 questions + some spacing
            
            for col in range(5):
                col_x = x + col * col_width + col_width // 2  # Center of column
                
                for row in range(20):
                    if question_num > self.total_questions:
                        break
                    
                    # Calculate base y position for this question
                    base_y = y + row * row_height + row_height // 2
                    
                    # Calculate positions for A, B, C, D options
                    # They should be vertically aligned
                    option_spacing = row_height // 5  # Space between options
                    
                    options = {}
                    for i, option in enumerate(['A', 'B', 'C', 'D']):
                        option_y = base_y + (i - 1.5) * option_spacing
                        options[option] = (int(col_x), int(option_y))
                    
                    bubble_positions[str(question_num)] = options
                    question_num += 1
            
            print(f"Built bubble positions for {len(bubble_positions)} questions")
            return bubble_positions
            
        except Exception as e:
            print(f"Error building bubble positions: {str(e)}")
            return {}
    
    def detect_filled_bubbles(self, image: np.ndarray, bubble_positions: Dict[str, Any], 
                            threshold: float) -> Dict[str, str]:
        """Detect which bubbles are filled"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            filled_bubbles = {}
            
            for question_num, options in bubble_positions.items():
                option_scores = {}
                
                for option, (x, y) in options.items():
                    # Extract bubble region
                    bubble_radius = 15  # Fixed radius
                    x1, y1 = max(0, x - bubble_radius), max(0, y - bubble_radius)
                    x2, y2 = min(gray.shape[1], x + bubble_radius), min(gray.shape[0], y + bubble_radius)
                    
                    bubble_region = gray[y1:y2, x1:x2]
                    
                    if bubble_region.size > 0:
                        # Calculate mean intensity (lower = darker = filled)
                        mean_intensity = np.mean(bubble_region)
                        option_scores[option] = mean_intensity
                
                # Find the darkest option
                if option_scores:
                    darkest_option = min(option_scores, key=option_scores.get)
                    darkest_score = option_scores[darkest_option]
                    
                    # Check if it's significantly darker than others
                    other_scores = [score for opt, score in option_scores.items() if opt != darkest_option]
                    if other_scores:
                        avg_other_score = np.mean(other_scores)
                        std_other_score = np.std(other_scores)
                        
                        # Use adaptive threshold
                        threshold_value = avg_other_score - max(15, std_other_score * 0.5)
                        
                        if darkest_score < threshold_value:
                            filled_bubbles[question_num] = darkest_option
                        else:
                            filled_bubbles[question_num] = None
                    else:
                        filled_bubbles[question_num] = None
                else:
                    filled_bubbles[question_num] = None
            
            filled_count = len([k for k, v in filled_bubbles.items() if v is not None])
            print(f"Detected filled bubbles for {filled_count} questions")
            return filled_bubbles
            
        except Exception as e:
            print(f"Error detecting filled bubbles: {str(e)}")
            return {}
    
    def evaluate_answers(self, filled_bubbles: Dict[str, str], answer_key: Dict[str, str], 
                        subjects: Dict[str, str]) -> Dict[str, Any]:
        """Evaluate answers and calculate scores"""
        results = {
            'student_id': None,
            'total_score': 0,
            'percentage': 0.0,
            'grade': 'D',
            'subjects': {},
            'per_question': {}
        }
        
        # Calculate subject scores
        subject_scores = {}
        for question_num in range(1, self.total_questions + 1):
            q_str = str(question_num)
            subject = subjects.get(q_str, 'Unknown')
            
            if subject not in subject_scores:
                subject_scores[subject] = {'correct': 0, 'total': 0}
            
            subject_scores[subject]['total'] += 1
            
            # Check if question was answered correctly
            detected = filled_bubbles.get(q_str)
            correct_answer = answer_key.get(q_str)
            
            is_correct = False
            is_valid = True
            
            if detected is not None and correct_answer is not None:
                is_correct = (detected == correct_answer)
            elif detected is not None:
                # Multiple options selected (invalid)
                is_valid = False
            
            if is_correct and is_valid:
                subject_scores[subject]['correct'] += 1
            
            # Store per-question result
            results['per_question'][q_str] = {
                'detected': detected if detected else 'None',
                'valid': is_valid,
                'correct': is_correct
            }
        
        # Calculate total score
        total_correct = sum(score['correct'] for score in subject_scores.values())
        results['total_score'] = total_correct
        results['percentage'] = (total_correct / self.total_questions) * 100
        
        # Calculate grade
        if results['percentage'] >= 85:
            results['grade'] = 'A'
        elif results['percentage'] >= 70:
            results['grade'] = 'B'
        elif results['percentage'] >= 50:
            results['grade'] = 'C'
        else:
            results['grade'] = 'D'
        
        # Format subject scores
        for subject, score in subject_scores.items():
            results['subjects'][subject] = {
                'score': score['correct'],
                'out_of': score['total']
            }
        
        return results
    
    def create_annotated_image(self, image: np.ndarray, bubble_positions: Dict[str, Any], 
                             filled_bubbles: Dict[str, str], results: Dict[str, Any]) -> np.ndarray:
        """Create annotated image showing results"""
        # Convert to color if grayscale
        if len(image.shape) == 2:
            annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            annotated = image.copy()
        
        # Define colors
        correct_color = (0, 255, 0)  # Green
        incorrect_color = (0, 0, 255)  # Red
        invalid_color = (0, 255, 255)  # Yellow
        unanswered_color = (128, 128, 128)  # Gray
        
        for question_num, options in bubble_positions.items():
            detected = filled_bubbles.get(question_num)
            question_result = results['per_question'].get(question_num, {})
            
            for option, (x, y) in options.items():
                # Draw bubble circle
                cv2.circle(annotated, (x, y), 15, (255, 255, 255), 2)
                
                # Color based on result
                if detected == option:
                    if question_result.get('correct', False):
                        color = correct_color
                    elif not question_result.get('valid', True):
                        color = invalid_color
                    else:
                        color = incorrect_color
                    
                    # Fill the bubble
                    cv2.circle(annotated, (x, y), 13, color, -1)
                    
                    # Add option label
                    cv2.putText(annotated, option, (x - 5, y + 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return annotated
