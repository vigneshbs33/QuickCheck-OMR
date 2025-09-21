"""
Adaptive OMR Engine for QuickCheck
Automatically adapts to different image sizes and orientations
Developed by InteliCat Team
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path


class AdaptiveOMRProcessor:
    """
    Adaptive OMR processor that automatically adjusts to image dimensions
    """
    
    def __init__(self):
        self.questions_per_column = 20
        self.options_per_question = 4
        self.total_questions = 100
        self.num_columns = 5
        
        # Reference template dimensions
        self.ref_width = 1200
        self.ref_height = 1122
        
        # Reference bubble positions (normalized to 0-1)
        self.ref_positions = self._build_reference_positions()
        
    def _build_reference_positions(self) -> Dict[str, Any]:
        """Build reference positions normalized to 0-1"""
        # Field blocks configuration from template
        field_blocks = {
            "MCQBlock1": {"origin": [50, 50], "fieldLabels": ["q1", "q2", "q3", "q4", "q5", "q6", "q7"], "bubblesGap": 25, "labelsGap": 25},
            "MCQBlock2": {"origin": [650, 50], "fieldLabels": ["q8", "q9", "q10", "q11", "q12", "q13", "q14"], "bubblesGap": 25, "labelsGap": 25},
            "MCQBlock3": {"origin": [50, 250], "fieldLabels": ["q15", "q16", "q17", "q18", "q19", "q20", "q21"], "bubblesGap": 25, "labelsGap": 25},
            "MCQBlock4": {"origin": [650, 250], "fieldLabels": ["q22", "q23", "q24", "q25", "q26", "q27", "q28"], "bubblesGap": 25, "labelsGap": 25},
            "MCQBlock5": {"origin": [50, 450], "fieldLabels": ["q29", "q30", "q31", "q32", "q33", "q34", "q35"], "bubblesGap": 25, "labelsGap": 25},
            "MCQBlock6": {"origin": [650, 450], "fieldLabels": ["q36", "q37", "q38", "q39", "q40", "q41", "q42"], "bubblesGap": 25, "labelsGap": 25},
            "MCQBlock7": {"origin": [50, 650], "fieldLabels": ["q43", "q44", "q45", "q46", "q47", "q48", "q49"], "bubblesGap": 25, "labelsGap": 25},
            "MCQBlock8": {"origin": [650, 650], "fieldLabels": ["q50", "q51", "q52", "q53", "q54", "q55", "q56"], "bubblesGap": 25, "labelsGap": 25},
            "MCQBlock9": {"origin": [50, 850], "fieldLabels": ["q57", "q58", "q59", "q60", "q61", "q62", "q63"], "bubblesGap": 25, "labelsGap": 25},
            "MCQBlock10": {"origin": [650, 850], "fieldLabels": ["q64", "q65", "q66", "q67", "q68", "q69", "q70"], "bubblesGap": 25, "labelsGap": 25},
            "MCQBlock11": {"origin": [50, 1020], "fieldLabels": ["q71", "q72", "q73", "q74", "q75", "q76", "q77"], "bubblesGap": 10, "labelsGap": 10},
            "MCQBlock12": {"origin": [650, 1020], "fieldLabels": ["q78", "q79", "q80", "q81", "q82", "q83", "q84"], "bubblesGap": 10, "labelsGap": 10},
            "MCQBlock13": {"origin": [50, 1040], "fieldLabels": ["q85", "q86", "q87", "q88", "q89", "q90", "q91"], "bubblesGap": 10, "labelsGap": 10},
            "MCQBlock14": {"origin": [650, 1020], "fieldLabels": ["q92", "q93", "q94", "q95", "q96", "q97", "q98"], "bubblesGap": 10, "labelsGap": 10},
            "MCQBlock15": {"origin": [50, 1080], "fieldLabels": ["q99", "q100"], "bubblesGap": 10, "labelsGap": 10}
        }
        
        bubble_positions = {}
        
        for block_name, block_config in field_blocks.items():
            origin_x, origin_y = block_config["origin"]
            field_labels = block_config["fieldLabels"]
            bubbles_gap = block_config["bubblesGap"]
            labels_gap = block_config["labelsGap"]
            
            for i, field_label in enumerate(field_labels):
                # Calculate y position for this question
                question_y = origin_y + i * labels_gap
                
                # Calculate x positions for A, B, C, D options
                option_x = origin_x + 50  # Start after question number
                
                # Create options dictionary with normalized coordinates
                options = {}
                for j, option in enumerate(['A', 'B', 'C', 'D']):
                    option_x_pos = option_x + j * bubbles_gap
                    # Normalize coordinates
                    norm_x = option_x_pos / self.ref_width
                    norm_y = question_y / self.ref_height
                    options[option] = (norm_x, norm_y)
                
                # Extract question number from field label (e.g., "q1" -> "1")
                question_num = field_label[1:]  # Remove 'q' prefix
                bubble_positions[question_num] = options
        
        return bubble_positions
    
    def process_omr(self, image: np.ndarray, answer_key: Dict[str, str], 
                   subjects: Dict[str, str], bubble_threshold: float = 0.4) -> Optional[Dict[str, Any]]:
        """
        Main processing function for OMR evaluation
        """
        try:
            print(f"Processing image of shape: {image.shape}")
            
            # Step 1: Detect and correct orientation
            corrected_image = self.correct_orientation(image)
            
            # Step 2: Scale positions to actual image dimensions
            bubble_positions = self.scale_positions_to_image(corrected_image)
            
            # Step 3: Detect filled bubbles
            filled_bubbles = self.detect_filled_bubbles(corrected_image, bubble_positions, bubble_threshold)
            
            # Step 4: Evaluate answers
            results = self.evaluate_answers(filled_bubbles, answer_key, subjects)
            
            # Step 5: Create annotated image
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
    
    def scale_positions_to_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Scale reference positions to actual image dimensions"""
        height, width = image.shape[:2]
        
        scaled_positions = {}
        
        for question_num, options in self.ref_positions.items():
            scaled_options = {}
            for option, (norm_x, norm_y) in options.items():
                # Scale to actual image dimensions
                actual_x = int(norm_x * width)
                actual_y = int(norm_y * height)
                scaled_options[option] = (actual_x, actual_y)
            scaled_positions[question_num] = scaled_options
        
        return scaled_positions
    
    def detect_filled_bubbles(self, image: np.ndarray, bubble_positions: Dict[str, Any], 
                            threshold: float) -> Dict[str, str]:
        """Detect filled bubbles using adaptive positions"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            filled_bubbles = {}
            
            for question_num, options in bubble_positions.items():
                option_scores = {}
                
                for option, (x, y) in options.items():
                    # Calculate adaptive bubble size based on image dimensions
                    image_area = gray.shape[0] * gray.shape[1]
                    bubble_radius = max(8, int(np.sqrt(image_area) / 60))  # Adaptive size
                    
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
                        
                        # Use adaptive threshold
                        threshold_value = avg_other_score - max(10, std_other_score * 0.3)
                        
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
                # Calculate adaptive bubble size
                image_area = image.shape[0] * image.shape[1]
                bubble_radius = max(8, int(np.sqrt(image_area) / 60))
                
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
