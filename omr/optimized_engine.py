"""
Optimized OMR Engine for QuickCheck
Based on comprehensive analysis of sample data
Developed by InteliCat Team
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
from sklearn.cluster import KMeans


class OptimizedOMRProcessor:
    """
    Highly optimized OMR processor based on analysis of sample data
    Achieves best performance on the specific OMR format
    """
    
    def __init__(self):
        self.questions_per_column = 20
        self.options_per_question = 4
        self.total_questions = 100
        self.num_columns = 5
        
        # Optimized parameters based on sample analysis
        self.bubble_detection_params = {
            'min_area_factor': 0.1,
            'max_area_factor': 3.0,
            'circularity_threshold': 0.3,
            'adaptive_threshold_factor': 0.3,
            'bubble_radius_factor': 60
        }
        
        # Calibration data
        self.calibrated_positions = None
        self.is_calibrated = False
        
    def calibrate_from_reference(self, reference_image: np.ndarray) -> bool:
        """Calibrate using a reference image for maximum accuracy"""
        try:
            print("Calibrating from reference image...")
            
            # Detect bubbles using optimized parameters
            bubble_centers = self.detect_bubbles_optimized(reference_image)
            if len(bubble_centers) < 200:
                print(f"Not enough bubbles found for calibration: {len(bubble_centers)}")
                return False
            
            # Build calibrated positions
            self.calibrated_positions = self.build_calibrated_positions(bubble_centers, reference_image.shape)
            if len(self.calibrated_positions) >= 50:
                self.is_calibrated = True
                print(f"Calibration successful! Found {len(self.calibrated_positions)} questions")
                return True
            else:
                print("Calibration failed - not enough questions found")
                return False
                
        except Exception as e:
            print(f"Error during calibration: {str(e)}")
            return False
    
    def detect_bubbles_optimized(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """Optimized bubble detection using multiple methods"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Try multiple thresholding methods
            bubble_centers = []
            
            # Method 1: Adaptive threshold
            thresh1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
            
            # Method 2: Otsu threshold
            _, thresh2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Method 3: Simple threshold
            _, thresh3 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Try each thresholding method
            for thresh in [thresh1, thresh2, thresh3]:
                centers = self._find_circles_optimized(thresh, image.shape)
                if len(centers) > len(bubble_centers):
                    bubble_centers = centers
            
            # If still not enough, try HoughCircles with optimized parameters
            if len(bubble_centers) < 200:
                circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
                                         param1=50, param2=30, minRadius=8, maxRadius=25)
                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    for (x, y, r) in circles:
                        bubble_centers.append((x, y))
            
            print(f"Detected {len(bubble_centers)} bubbles for calibration")
            return bubble_centers
            
        except Exception as e:
            print(f"Error detecting bubbles: {str(e)}")
            return []
    
    def _find_circles_optimized(self, thresh: np.ndarray, image_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Optimized circle finding with learned parameters"""
        bubble_centers = []
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate expected bubble size based on image dimensions
        image_area = image_shape[0] * image_shape[1]
        expected_bubble_area = image_area / 4000  # Rough estimate for 100 questions * 4 options
        min_area = expected_bubble_area * self.bubble_detection_params['min_area_factor']
        max_area = expected_bubble_area * self.bubble_detection_params['max_area_factor']
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > self.bubble_detection_params['circularity_threshold']:
                        # Get center
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            bubble_centers.append((cx, cy))
        
        return bubble_centers
    
    def build_calibrated_positions(self, bubble_centers: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Build calibrated positions using advanced clustering"""
        try:
            # Convert to numpy array
            centers = np.array(bubble_centers)
            
            # Cluster by x-coordinate to find columns
            kmeans_x = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans_x.fit(centers[:, 0].reshape(-1, 1))
            column_labels = kmeans_x.labels_
            
            # Group bubbles by column
            column_clusters = []
            for i in range(5):
                col_bubbles = [bubble_centers[j] for j in range(len(bubble_centers)) if column_labels[j] == i]
                if len(col_bubbles) >= 20:  # Need at least 20 bubbles per column
                    column_clusters.append(sorted(col_bubbles, key=lambda x: x[1]))  # Sort by y-coordinate
            
            if len(column_clusters) != 5:
                print(f"Expected 5 columns, found {len(column_clusters)}")
                return {}
            
            # Build grid structure
            grid = {}
            question_num = 1
            
            for col_idx, col_bubbles in enumerate(column_clusters):
                print(f"Column {col_idx + 1}: {len(col_bubbles)} bubbles")
                
                # Group bubbles by rows (questions) - 4 bubbles per question
                questions_in_column = []
                
                # Use K-means to cluster by y-coordinate for better grouping
                if len(col_bubbles) >= 20:
                    y_coords = np.array([b[1] for b in col_bubbles]).reshape(-1, 1)
                    kmeans_y = KMeans(n_clusters=20, random_state=42, n_init=10)
                    kmeans_y.fit(y_coords)
                    row_labels = kmeans_y.labels_
                    
                    # Group bubbles by row
                    for row in range(20):
                        row_bubbles = [col_bubbles[j] for j in range(len(col_bubbles)) if row_labels[j] == row]
                        if len(row_bubbles) >= 4:
                            # Sort by y-coordinate to get A, B, C, D order
                            row_bubbles.sort(key=lambda x: x[1])
                            questions_in_column.append(row_bubbles[:4])  # Take first 4
                
                # Map to question numbers
                for q_idx, question_bubbles in enumerate(questions_in_column[:20]):  # Max 20 questions per column
                    if question_num <= self.total_questions and len(question_bubbles) == 4:
                        grid[str(question_num)] = {
                            'A': question_bubbles[0],
                            'B': question_bubbles[1],
                            'C': question_bubbles[2],
                            'D': question_bubbles[3]
                        }
                        question_num += 1
            
            print(f"Built calibrated grid with {len(grid)} questions")
            return grid
            
        except Exception as e:
            print(f"Error building calibrated positions: {str(e)}")
            return {}
    
    def process_omr(self, image: np.ndarray, answer_key: Dict[str, str], 
                   subjects: Dict[str, str], bubble_threshold: float = 0.4) -> Optional[Dict[str, Any]]:
        """
        Main processing function for OMR evaluation
        """
        try:
            print(f"Processing image of shape: {image.shape}")
            
            # If not calibrated, try to calibrate from this image
            if not self.is_calibrated:
                if not self.calibrate_from_reference(image):
                    print("Failed to calibrate from image")
                    return None
            
            if not self.calibrated_positions:
                print("No calibrated positions available")
                return None
            
            # Detect filled bubbles using calibrated positions
            filled_bubbles = self.detect_filled_bubbles_optimized(image, bubble_threshold)
            
            # Evaluate answers
            results = self.evaluate_answers(filled_bubbles, answer_key, subjects)
            
            # Create annotated image
            annotated_image = self.create_annotated_image(image, filled_bubbles, results)
            results['annotated_image'] = annotated_image
            
            return results
            
        except Exception as e:
            print(f"Error in OMR processing: {str(e)}")
            return None
    
    def detect_filled_bubbles_optimized(self, image: np.ndarray, threshold: float) -> Dict[str, str]:
        """Optimized filled bubble detection using calibrated positions"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            filled_bubbles = {}
            
            for question_num, options in self.calibrated_positions.items():
                option_scores = {}
                
                for option, (x, y) in options.items():
                    # Calculate adaptive bubble size
                    image_area = gray.shape[0] * gray.shape[1]
                    bubble_radius = max(8, int(np.sqrt(image_area) / self.bubble_detection_params['bubble_radius_factor']))
                    
                    x1, y1 = max(0, x - bubble_radius), max(0, y - bubble_radius)
                    x2, y2 = min(gray.shape[1], x + bubble_radius), min(gray.shape[0], y + bubble_radius)
                    
                    bubble_region = gray[y1:y2, x1:x2]
                    
                    if bubble_region.size > 0:
                        # Calculate multiple metrics
                        mean_intensity = np.mean(bubble_region)
                        min_intensity = np.min(bubble_region)
                        std_intensity = np.std(bubble_region)
                        
                        # Combined score (lower is more likely filled)
                        combined_score = mean_intensity * 0.7 + min_intensity * 0.2 + std_intensity * 0.1
                        option_scores[option] = combined_score
                
                # Find the darkest option
                if option_scores:
                    darkest_option = min(option_scores, key=option_scores.get)
                    darkest_score = option_scores[darkest_option]
                    
                    # Check if it's significantly darker than others
                    other_scores = [score for opt, score in option_scores.items() if opt != darkest_option]
                    if other_scores:
                        avg_other_score = np.mean(other_scores)
                        std_other_score = np.std(other_scores)
                        
                        # Use optimized threshold
                        threshold_value = avg_other_score - max(10, std_other_score * self.bubble_detection_params['adaptive_threshold_factor'])
                        
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
    
    def create_annotated_image(self, image: np.ndarray, filled_bubbles: Dict[str, str], 
                             results: Dict[str, Any]) -> np.ndarray:
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
        
        for question_num, options in self.calibrated_positions.items():
            detected = filled_bubbles.get(question_num)
            question_result = results['per_question'].get(question_num, {})
            
            for option, (x, y) in options.items():
                # Calculate adaptive bubble size
                image_area = image.shape[0] * image.shape[1]
                bubble_radius = max(8, int(np.sqrt(image_area) / self.bubble_detection_params['bubble_radius_factor']))
                
                # Draw bubble circle
                cv2.circle(annotated, (x, y), bubble_radius, (255, 255, 255), 2)
                
                # Color based on result
                if detected == option:
                    if question_result.get('correct', False):
                        color = correct_color
                    elif not question_result.get('valid', True):
                        color = invalid_color
                    else:
                        color = incorrect_color
                    
                    # Fill the bubble
                    cv2.circle(annotated, (x, y), bubble_radius - 2, color, -1)
                    
                    # Add option label
                    cv2.putText(annotated, option, (x - 5, y + 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated
