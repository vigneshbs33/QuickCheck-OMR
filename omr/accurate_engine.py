"""
Accurate OMR Engine for QuickCheck
Uses the exact template structure provided for maximum accuracy
Developed by InteliCat Team
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path


class AccurateOMRProcessor:
    """
    Highly accurate OMR processor using the exact template structure
    Based on the provided template with precise bubble positions
    """
    
    def __init__(self):
        self.questions_per_column = 20
        self.options_per_question = 4
        self.total_questions = 100
        self.num_columns = 5
        
        # Template configuration
        self.page_dimensions = [1200, 1122]
        self.bubble_dimensions = [20, 20]
        
        # Field blocks configuration
        self.field_blocks = {
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
        
        # Build bubble positions from template
        self.bubble_positions = self._build_bubble_positions()
        
    def _build_bubble_positions(self) -> Dict[str, Any]:
        """Build exact bubble positions from template configuration"""
        bubble_positions = {}
        
        for block_name, block_config in self.field_blocks.items():
            origin_x, origin_y = block_config["origin"]
            field_labels = block_config["fieldLabels"]
            bubbles_gap = block_config["bubblesGap"]
            labels_gap = block_config["labelsGap"]
            
            for i, field_label in enumerate(field_labels):
                # Calculate y position for this question
                question_y = origin_y + i * labels_gap
                
                # Calculate x positions for A, B, C, D options
                option_x = origin_x + 50  # Start after question number
                
                # Create options dictionary
                options = {}
                for j, option in enumerate(['A', 'B', 'C', 'D']):
                    option_x_pos = option_x + j * bubbles_gap
                    options[option] = (int(option_x_pos), int(question_y))
                
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
            
            # Step 1: Resize image to template dimensions
            resized_image = self._resize_to_template(image)
            
            # Step 2: Detect filled bubbles using exact positions
            filled_bubbles = self.detect_filled_bubbles(resized_image, bubble_threshold)
            
            # Step 3: Evaluate answers
            results = self.evaluate_answers(filled_bubbles, answer_key, subjects)
            
            # Step 4: Create annotated image
            annotated_image = self.create_annotated_image(resized_image, filled_bubbles, results)
            results['annotated_image'] = annotated_image
            
            return results
            
        except Exception as e:
            print(f"Error in OMR processing: {str(e)}")
            return None
    
    def _resize_to_template(self, image: np.ndarray) -> np.ndarray:
        """Resize image to match template dimensions"""
        target_width, target_height = self.page_dimensions
        
        # Resize image to template dimensions
        resized = cv2.resize(image, (target_width, target_height))
        
        return resized
    
    def detect_filled_bubbles(self, image: np.ndarray, threshold: float) -> Dict[str, str]:
        """Detect filled bubbles using exact template positions"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            filled_bubbles = {}
            
            for question_num, options in self.bubble_positions.items():
                option_scores = {}
                
                for option, (x, y) in options.items():
                    # Extract bubble region
                    bubble_radius = self.bubble_dimensions[0] // 2
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
        
        for question_num, options in self.bubble_positions.items():
            detected = filled_bubbles.get(question_num)
            question_result = results['per_question'].get(question_num, {})
            
            for option, (x, y) in options.items():
                # Draw bubble circle
                cv2.circle(annotated, (x, y), self.bubble_dimensions[0] // 2, (255, 255, 255), 2)
                
                # Color based on result
                if detected == option:
                    if question_result.get('correct', False):
                        color = correct_color
                    elif not question_result.get('valid', True):
                        color = invalid_color
                    else:
                        color = incorrect_color
                    
                    # Fill the bubble
                    cv2.circle(annotated, (x, y), self.bubble_dimensions[0] // 2 - 2, color, -1)
                    
                    # Add option label
                    cv2.putText(annotated, option, (x - 5, y + 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated
