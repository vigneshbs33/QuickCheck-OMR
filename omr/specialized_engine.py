"""
Specialized OMR Engine for QuickCheck
Optimized for the specific OMR format with 5 columns and 20 questions per column
Developed by InteliCat Team
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path


class SpecializedOMRProcessor:
    """
    Specialized OMR processor optimized for the specific format:
    - 5 columns (Python, Data Analysis, MySQL, Power BI, Adv Stats)
    - 20 questions per column
    - 4 options (A, B, C, D) per question
    - Total 100 questions
    """
    
    def __init__(self):
        self.questions_per_column = 20
        self.options_per_question = 4
        self.total_questions = 100
        self.num_columns = 5
        
    def process_omr(self, image: np.ndarray, answer_key: Dict[str, str], 
                   subjects: Dict[str, str], bubble_threshold: float = 0.4) -> Optional[Dict[str, Any]]:
        """
        Main processing function for OMR evaluation
        """
        try:
            print(f"Processing image of shape: {image.shape}")
            
            # Step 1: Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Step 2: Detect the OMR grid area
            grid_area = self.detect_grid_area(processed_image)
            if grid_area is None:
                print("Could not detect grid area")
                return None
            
            # Step 3: Extract and process each column
            columns = self.extract_columns(processed_image, grid_area)
            if len(columns) != 5:
                print(f"Expected 5 columns, found {len(columns)}")
                return None
            
            # Step 4: Process each column to find bubbles
            grid = self.build_grid_from_columns(processed_image, columns)
            if len(grid) < 50:  # Need at least half the questions
                print(f"Could not build proper grid, only {len(grid)} questions found")
                return None
            
            # Step 5: Detect filled bubbles
            filled_bubbles = self.detect_filled_bubbles(processed_image, grid, bubble_threshold)
            
            # Step 6: Evaluate answers
            results = self.evaluate_answers(filled_bubbles, answer_key, subjects)
            
            # Step 7: Create annotated image
            annotated_image = self.create_annotated_image(processed_image, grid, filled_bubbles, results)
            results['annotated_image'] = annotated_image
            
            return results
            
        except Exception as e:
            print(f"Error in OMR processing: {str(e)}")
            return None
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better bubble detection"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        return thresh
    
    def detect_grid_area(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect the main OMR grid area"""
        try:
            # Find contours
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the largest rectangular area
            max_area = 0
            best_contour = None
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    # Check if it's roughly rectangular
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if len(approx) >= 4:  # At least 4 corners
                        max_area = area
                        best_contour = contour
            
            if best_contour is not None:
                x, y, w, h = cv2.boundingRect(best_contour)
                return (x, y, w, h)
            
            return None
            
        except Exception as e:
            print(f"Error detecting grid area: {str(e)}")
            return None
    
    def extract_columns(self, image: np.ndarray, grid_area: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
        """Extract the 5 columns from the grid area"""
        x, y, w, h = grid_area
        
        # Calculate column width
        col_width = w // 5
        
        columns = []
        for i in range(5):
            col_x = x + i * col_width
            col_y = y
            col_w = col_width
            col_h = h
            columns.append((col_x, col_y, col_w, col_h))
        
        return columns
    
    def build_grid_from_columns(self, image: np.ndarray, columns: List[Tuple[int, int, int, int]]) -> Dict[str, Any]:
        """Build grid by processing each column"""
        grid = {}
        question_num = 1
        
        for col_idx, (col_x, col_y, col_w, col_h) in enumerate(columns):
            print(f"Processing column {col_idx + 1}")
            
            # Extract column region
            col_region = image[col_y:col_y+col_h, col_x:col_x+col_w]
            
            # Find bubbles in this column
            bubbles = self.find_bubbles_in_column(col_region, col_x, col_y)
            
            # Group bubbles by questions (4 bubbles per question)
            questions = self.group_bubbles_by_questions(bubbles, 20)  # 20 questions per column
            
            # Add to grid
            for q_idx, question_bubbles in enumerate(questions):
                if question_num <= self.total_questions and len(question_bubbles) == 4:
                    # Sort by y-coordinate to get A, B, C, D order
                    question_bubbles.sort(key=lambda b: b[1])
                    grid[str(question_num)] = {
                        'A': question_bubbles[0],
                        'B': question_bubbles[1],
                        'C': question_bubbles[2],
                        'D': question_bubbles[3]
                    }
                    question_num += 1
        
        return grid
    
    def find_bubbles_in_column(self, col_region: np.ndarray, offset_x: int, offset_y: int) -> List[Tuple[int, int]]:
        """Find bubbles in a specific column"""
        bubbles = []
        
        # Find contours
        contours, _ = cv2.findContours(col_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate expected bubble size based on column dimensions
        col_area = col_region.shape[0] * col_region.shape[1]
        expected_bubble_area = col_area / 80  # Rough estimate for 20 questions * 4 options
        min_area = expected_bubble_area * 0.2
        max_area = expected_bubble_area * 3
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.3:  # Less strict requirement
                        # Get center
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"]) + offset_x
                            cy = int(M["m01"] / M["m00"]) + offset_y
                            bubbles.append((cx, cy))
        
        print(f"Found {len(bubbles)} bubbles in column")
        return bubbles
    
    def group_bubbles_by_questions(self, bubbles: List[Tuple[int, int]], num_questions: int) -> List[List[Tuple[int, int]]]:
        """Group bubbles by questions using clustering"""
        if len(bubbles) < num_questions * 4:
            return []
        
        # Sort bubbles by y-coordinate
        bubbles.sort(key=lambda b: b[1])
        
        # Group into questions (4 bubbles per question)
        questions = []
        for i in range(0, len(bubbles), 4):
            if i + 4 <= len(bubbles):
                question_bubbles = bubbles[i:i+4]
                # Sort by y-coordinate within question
                question_bubbles.sort(key=lambda b: b[1])
                questions.append(question_bubbles)
        
        return questions[:num_questions]  # Limit to expected number of questions
    
    def detect_filled_bubbles(self, image: np.ndarray, grid: Dict[str, Any], threshold: float) -> Dict[str, str]:
        """Detect which bubbles are filled"""
        # Convert to grayscale for intensity analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        filled_bubbles = {}
        
        for question_num, options in grid.items():
            option_scores = {}
            
            for option, (x, y) in options.items():
                # Extract bubble region
                bubble_radius = 12  # Fixed radius for consistency
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
                    if darkest_score < avg_other_score - 10:  # Threshold for filled detection
                        filled_bubbles[question_num] = darkest_option
                    else:
                        filled_bubbles[question_num] = None
                else:
                    filled_bubbles[question_num] = None
            else:
                filled_bubbles[question_num] = None
        
        print(f"Detected filled bubbles for {len([k for k, v in filled_bubbles.items() if v is not None])} questions")
        return filled_bubbles
    
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
    
    def create_annotated_image(self, image: np.ndarray, grid: Dict[str, Any], 
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
        
        for question_num, options in grid.items():
            detected = filled_bubbles.get(question_num)
            question_result = results['per_question'].get(question_num, {})
            
            for option, (x, y) in options.items():
                # Draw bubble circle
                cv2.circle(annotated, (x, y), 12, (255, 255, 255), 2)
                
                # Color based on result
                if detected == option:
                    if question_result.get('correct', False):
                        color = correct_color
                    elif not question_result.get('valid', True):
                        color = invalid_color
                    else:
                        color = incorrect_color
                    
                    # Fill the bubble
                    cv2.circle(annotated, (x, y), 10, color, -1)
                    
                    # Add option label
                    cv2.putText(annotated, option, (x - 5, y + 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated
