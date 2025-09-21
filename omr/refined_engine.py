"""
Refined OMR Engine for QuickCheck
Balanced approach with good accuracy and performance
Developed by InteliCat Team
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
from sklearn.cluster import KMeans
import statistics
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RefinedOMRProcessor:
    """
    Refined OMR processor with balanced accuracy and performance
    Optimized for Innomatics-style OMR sheets
    """
    
    def __init__(self):
        self.questions_per_column = 20
        self.options_per_question = 4
        self.total_questions = 100
        self.num_columns = 5
        
        # Refined parameters
        self.detection_params = {
            'min_area': 20,           # Balanced minimum
            'max_area': 200,          # Balanced maximum
            'min_circularity': 0.3,   # Balanced threshold
            'adaptive_threshold_factor': 0.3,  # Balanced sensitivity
            'bubble_radius_factor': 60,
            'gaussian_kernel': (3, 3),
            'morphology_kernel_size': 2,
            'fill_ratio_threshold': 0.2,  # Balanced threshold
            'contrast_enhancement': 1.3,
            'brightness_adjustment': 5,
            'edge_threshold1': 40,
            'edge_threshold2': 120
        }
        
        # Calibration data
        self.calibrated_positions = None
        self.is_calibrated = False
        self.reference_image_shape = None
        self.bubble_diameter = 0
        
        # Subject mapping
        self.subject_ranges = {
            'PYTHON': {'start': 1, 'end': 20},
            'DATA ANALYSIS': {'start': 21, 'end': 40},
            'MySQL': {'start': 41, 'end': 60},
            'POWER BI': {'start': 61, 'end': 80},
            'Adv STATS': {'start': 81, 'end': 100}
        }
    
    def calibrate_from_reference(self, reference_image: np.ndarray) -> bool:
        """Refined calibration with balanced approach"""
        try:
            logger.info("Starting refined calibration...")
            
            # Balanced preprocessing
            processed_image = self._refined_preprocess_image(reference_image)
            
            # Balanced bubble detection
            bubble_centers = self._detect_bubbles_refined(processed_image, reference_image.shape)
            
            if len(bubble_centers) < 150:  # Balanced threshold
                logger.warning(f"Not enough bubbles found for calibration: {len(bubble_centers)}")
                return False
            
            # Build refined positions
            self.calibrated_positions = self._build_refined_positions(bubble_centers, reference_image.shape)
            
            if len(self.calibrated_positions) >= 40:  # Balanced threshold
                self.is_calibrated = True
                self.reference_image_shape = reference_image.shape
                logger.info(f"Refined calibration successful! Found {len(self.calibrated_positions)} questions")
                return True
            else:
                logger.warning("Calibration failed - not enough questions found")
                return False
                
        except Exception as e:
            logger.error(f"Error during refined calibration: {str(e)}")
            return False
    
    def _refined_preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Refined preprocessing with balanced approach"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 1. Light noise reduction
        denoised = cv2.bilateralFilter(gray, 5, 30, 30)
        
        # 2. Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 3. Brightness adjustment
        enhanced = cv2.convertScaleAbs(enhanced, alpha=self.detection_params['contrast_enhancement'], 
                                     beta=self.detection_params['brightness_adjustment'])
        
        # 4. Gaussian blur
        blurred = cv2.GaussianBlur(enhanced, self.detection_params['gaussian_kernel'], 0)
        
        # 5. Adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # 6. Edge detection
        edges = cv2.Canny(blurred, self.detection_params['edge_threshold1'], 
                         self.detection_params['edge_threshold2'])
        
        # 7. Combine strategies
        combined = cv2.bitwise_or(thresh, edges)
        
        # 8. Light morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (self.detection_params['morphology_kernel_size'], 
                                          self.detection_params['morphology_kernel_size']))
        processed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        return processed
    
    def _detect_bubbles_refined(self, processed: np.ndarray, image_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Refined bubble detection with balanced approach"""
        bubble_centers = []
        
        # Method 1: Contour detection
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.detection_params['min_area'] < area < self.detection_params['max_area']:
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > self.detection_params['min_circularity']:
                        # Calculate moments for center
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            bubble_centers.append((cx, cy))
        
        # Method 2: HoughCircles
        circles = cv2.HoughCircles(processed, cv2.HOUGH_GRADIENT, 1, 20,
                                 param1=50, param2=30, minRadius=4, maxRadius=20)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                if 4 <= r <= 20:
                    bubble_centers.append((x, y))
        
        # Remove duplicates
        bubble_centers = list(set(bubble_centers))
        
        logger.info(f"Refined detection found {len(bubble_centers)} bubbles")
        return bubble_centers
    
    def _build_refined_positions(self, bubble_centers: List[Tuple[int, int]], 
                               image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Build refined positions with balanced clustering"""
        try:
            centers = np.array(bubble_centers)
            
            # Column detection
            x_coords = centers[:, 0].reshape(-1, 1)
            
            # Use 5 columns for Innomatics format
            kmeans_x = KMeans(n_clusters=5, random_state=42, n_init=20)
            kmeans_x.fit(x_coords)
            column_labels = kmeans_x.labels_
            
            # Group bubbles by column
            column_clusters = []
            for i in range(5):
                col_bubbles = [bubble_centers[j] for j in range(len(bubble_centers)) if column_labels[j] == i]
                if len(col_bubbles) >= 8:  # Balanced threshold
                    column_clusters.append(sorted(col_bubbles, key=lambda x: x[1]))
            
            if len(column_clusters) < 4:
                logger.warning(f"Expected at least 4 columns, found {len(column_clusters)}")
                return {}
            
            # Build refined grid
            grid = {}
            question_num = 1
            
            for col_idx, col_bubbles in enumerate(column_clusters):
                logger.info(f"Column {col_idx + 1}: {len(col_bubbles)} bubbles")
                
                # Row clustering
                if len(col_bubbles) >= 10:
                    y_coords = np.array([b[1] for b in col_bubbles]).reshape(-1, 1)
                    
                    # Use 20 rows for Innomatics format
                    kmeans_y = KMeans(n_clusters=20, random_state=42, n_init=20)
                    kmeans_y.fit(y_coords)
                    row_labels = kmeans_y.labels_
                    
                    # Group bubbles by row
                    questions_in_column = []
                    for row in range(20):
                        row_bubbles = [col_bubbles[j] for j in range(len(col_bubbles)) if row_labels[j] == row]
                        if len(row_bubbles) >= 2:  # At least 2 options
                            row_bubbles.sort(key=lambda x: x[0])  # Sort by x-coordinate
                            # Pad to 4 options if needed
                            while len(row_bubbles) < 4:
                                row_bubbles.append(row_bubbles[-1] if row_bubbles else (0, 0))
                            questions_in_column.append(row_bubbles[:4])
                    
                    # Map to question numbers
                    for q_idx, question_bubbles in enumerate(questions_in_column[:20]):
                        if question_num <= self.total_questions and len(question_bubbles) == 4:
                            grid[str(question_num)] = {
                                'A': question_bubbles[0],
                                'B': question_bubbles[1],
                                'C': question_bubbles[2],
                                'D': question_bubbles[3]
                            }
                            question_num += 1
            
            logger.info(f"Built refined grid with {len(grid)} questions")
            return grid
            
        except Exception as e:
            logger.error(f"Error building refined positions: {str(e)}")
            return {}
    
    def process_omr_sheet(self, image: np.ndarray, answer_key: Dict[str, str], 
                         subjects: Dict[str, Dict[str, int]], bubble_threshold: float = 0.2,
                         grade_thresholds: Dict[str, int] = None) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
        """Main processing function with refined approach"""
        try:
            logger.info(f"Processing image with refined approach: {image.shape}")
            
            # Calibrate if not already calibrated
            if not self.is_calibrated:
                if not self.calibrate_from_reference(image):
                    logger.error("Failed to calibrate from image")
                    return None, None
            
            if not self.calibrated_positions:
                logger.error("No calibrated positions available")
                return None, None
            
            # Detect filled bubbles using refined approach
            filled_bubbles = self._detect_filled_bubbles_refined(image, bubble_threshold)
            
            # Evaluate answers
            results = self._evaluate_answers_refined(filled_bubbles, answer_key, subjects, grade_thresholds)
            
            # Create annotated image
            annotated_image = self._create_annotated_image_refined(image, filled_bubbles, results)
            
            return results, annotated_image
            
        except Exception as e:
            logger.error(f"Error in refined OMR processing: {str(e)}")
            return None, None
    
    def _detect_filled_bubbles_refined(self, image: np.ndarray, threshold: float) -> Dict[str, str]:
        """Refined filled bubble detection"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            processed = self._refined_preprocess_image(image)
            
            filled_bubbles = {}
            
            for question_num, options in self.calibrated_positions.items():
                option_scores = {}
                
                for option, (x, y) in options.items():
                    # Calculate adaptive bubble size
                    bubble_radius = self._calculate_adaptive_bubble_size(image.shape)
                    
                    x1, y1 = max(0, x - bubble_radius), max(0, y - bubble_radius)
                    x2, y2 = min(gray.shape[1], x + bubble_radius), min(gray.shape[0], y + bubble_radius)
                    
                    bubble_region = processed[y1:y2, x1:x2]
                    
                    if bubble_region.size > 0:
                        # Calculate metrics
                        mean_intensity = np.mean(bubble_region)
                        min_intensity = np.min(bubble_region)
                        std_intensity = np.std(bubble_region)
                        
                        # Calculate fill ratio
                        total_pixels = bubble_region.size
                        filled_pixels = np.sum(bubble_region > 0)
                        fill_ratio = filled_pixels / total_pixels if total_pixels > 0 else 0
                        
                        # Refined combined score
                        combined_score = (fill_ratio * 0.5 + 
                                        (255 - mean_intensity) / 255 * 0.3 +
                                        (255 - min_intensity) / 255 * 0.15 +
                                        std_intensity / 255 * 0.05)
                        
                        option_scores[option] = combined_score
                
                # Refined filled detection
                if option_scores:
                    # Find the darkest option
                    darkest_option = min(option_scores, key=option_scores.get)
                    darkest_score = option_scores[darkest_option]
                    
                    # Calculate refined threshold
                    other_scores = [score for opt, score in option_scores.items() if opt != darkest_option]
                    if other_scores:
                        avg_other_score = np.mean(other_scores)
                        std_other_score = np.std(other_scores)
                        
                        # Refined adaptive threshold
                        threshold_value = (avg_other_score - 
                                         max(0.05, std_other_score * self.detection_params['adaptive_threshold_factor']))
                        
                        if darkest_score < threshold_value:
                            filled_bubbles[question_num] = darkest_option
                        else:
                            filled_bubbles[question_num] = None
                    else:
                        filled_bubbles[question_num] = None
                else:
                    filled_bubbles[question_num] = None
            
            filled_count = len([k for k, v in filled_bubbles.items() if v is not None])
            logger.info(f"Refined detection found {filled_count} filled bubbles")
            return filled_bubbles
            
        except Exception as e:
            logger.error(f"Error in refined filled bubble detection: {str(e)}")
            return {}
    
    def _calculate_adaptive_bubble_size(self, image_shape: Tuple[int, int]) -> int:
        """Calculate adaptive bubble size based on image dimensions"""
        image_area = image_shape[0] * image_shape[1]
        return max(5, int(np.sqrt(image_area) / self.detection_params['bubble_radius_factor']))
    
    def _evaluate_answers_refined(self, filled_bubbles: Dict[str, str], answer_key: Dict[str, str], 
                                 subjects: Dict[str, Dict[str, int]], 
                                 grade_thresholds: Dict[str, int] = None) -> Dict[str, Any]:
        """Refined answer evaluation"""
        if grade_thresholds is None:
            grade_thresholds = {'A': 85, 'B': 70, 'C': 50, 'D': 0}
        
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
            
            # Find subject for this question
            subject = 'Unknown'
            for subj_name, subj_data in subjects.items():
                if subj_data['start'] <= question_num <= subj_data['end']:
                    subject = subj_name
                    break
            
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
        if results['percentage'] >= grade_thresholds['A']:
            results['grade'] = 'A'
        elif results['percentage'] >= grade_thresholds['B']:
            results['grade'] = 'B'
        elif results['percentage'] >= grade_thresholds['C']:
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
    
    def _create_annotated_image_refined(self, image: np.ndarray, filled_bubbles: Dict[str, str], 
                                      results: Dict[str, Any]) -> np.ndarray:
        """Create refined annotated image"""
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
                bubble_radius = self._calculate_adaptive_bubble_size(image.shape)
                
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

