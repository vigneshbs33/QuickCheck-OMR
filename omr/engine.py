"""
QuickCheck OMR Engine
Developed by InteliCat Team

This implementation is provided to InteliCat for exclusive use under the project name QuickCheck.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path


class OMRProcessor:
    """
    Main OMR processing engine that handles:
    - Sheet detection and orientation correction
    - Template-free bubble detection
    - Answer evaluation and scoring
    """
    
    def __init__(self):
        self.bubble_diameter = 25  # Expected bubble diameter in pixels
        self.column_centers = [177, 386, 563, 783, 977]  # Expected column centers for 1100px width
        self.questions_per_column = 20
        self.options_per_question = 4
        self.total_questions = 100
        
    def process_omr(self, image: np.ndarray, answer_key: Dict[str, str], 
                   subjects: Dict[str, str], bubble_threshold: float = 0.4) -> Optional[Dict[str, Any]]:
        """
        Main processing function for OMR evaluation
        
        Args:
            image: Input OMR sheet image
            answer_key: Dictionary mapping question numbers to correct answers
            subjects: Dictionary mapping question numbers to subject names
            bubble_threshold: Threshold for detecting filled bubbles (0.0-1.0)
            
        Returns:
            Dictionary containing evaluation results or None if processing fails
        """
        try:
            # Step 1: Detect and correct sheet orientation
            corrected_image = self.detect_sheet(image)
            if corrected_image is None:
                return None
            
            # Step 2: Detect bubble centers using clustering
            bubble_centers = self.detect_bubbles(corrected_image)
            if len(bubble_centers) < 400:  # Should have at least 100 questions * 4 options
                return None
            
            # Step 3: Build grid from detected bubbles
            grid = self.build_grid(bubble_centers, corrected_image.shape)
            if grid is None:
                return None
            
            # Step 4: Detect filled bubbles
            filled_bubbles = self.detect_filled(corrected_image, grid, bubble_threshold)
            
            # Step 5: Evaluate answers
            results = self.evaluate(filled_bubbles, answer_key, subjects)
            
            # Step 6: Create annotated image
            annotated_image = self.create_annotated_image(corrected_image, grid, filled_bubbles, results)
            results['annotated_image'] = annotated_image
            
            return results
            
        except Exception as e:
            print(f"Error in OMR processing: {str(e)}")
            return None
    
    def detect_sheet(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect OMR sheet and correct orientation using contour detection
        """
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
            
            # Find the largest rectangular contour (likely the OMR sheet)
            largest_contour = None
            max_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    # Approximate the contour
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Check if it's roughly rectangular (4 corners)
                    if len(approx) == 4 and area > max_area:
                        max_area = area
                        largest_contour = approx
            
            if largest_contour is not None:
                # Apply perspective transform to correct orientation
                return self.apply_perspective_transform(image, largest_contour.reshape(4, 2))
            else:
                # If no rectangular contour found, try to detect rotation
                return self.detect_rotation(image)
                
        except Exception as e:
            print(f"Error in sheet detection: {str(e)}")
            return None
    
    def apply_perspective_transform(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Apply perspective transform to correct sheet orientation"""
        # Order points: top-left, top-right, bottom-right, bottom-left
        rect = self.order_points(corners)
        
        # Calculate dimensions
        width_a = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
        width_b = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        
        height_a = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
        height_b = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        
        # Destination points
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)
        
        # Apply transformation
        matrix = cv2.getPerspectiveTransform(rect.astype(np.float32), dst)
        warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
        
        return warped
    
    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points in consistent order: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Top-left point has smallest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        
        # Bottom-right point has largest sum
        rect[2] = pts[np.argmax(s)]
        
        # Top-right point has smallest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        
        # Bottom-left point has largest difference
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def detect_rotation(self, image: np.ndarray) -> np.ndarray:
        """Detect and correct rotation using Hough lines"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
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
            print(f"Error in rotation detection: {str(e)}")
            return image
    
    def detect_bubbles(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect bubble centers using contour detection and clustering
        """
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
                centers = self._find_circles_in_thresh(thresh, image.shape)
                if len(centers) > len(bubble_centers):
                    bubble_centers = centers
            
            # If still not enough bubbles, try HoughCircles
            if len(bubble_centers) < 200:
                circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
                                         param1=50, param2=30, minRadius=8, maxRadius=25)
                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    for (x, y, r) in circles:
                        bubble_centers.append((x, y))
            
            print(f"Detected {len(bubble_centers)} bubbles")
            return bubble_centers
            
        except Exception as e:
            print(f"Error in bubble detection: {str(e)}")
            return []
    
    def _find_circles_in_thresh(self, thresh: np.ndarray, image_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find circles in thresholded image"""
        bubble_centers = []
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate expected bubble size based on image dimensions
        image_area = image_shape[0] * image_shape[1]
        expected_bubble_area = image_area / 4000  # Rough estimate for 100 questions * 4 options
        min_area = expected_bubble_area * 0.3
        max_area = expected_bubble_area * 3
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.5:  # Less strict circularity requirement
                        # Get center
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            bubble_centers.append((cx, cy))
        
        return bubble_centers
    
    def build_grid(self, bubble_centers: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        """
        Build a grid structure from detected bubble centers using clustering
        """
        try:
            if len(bubble_centers) < 100:  # Need at least 25 questions worth of bubbles
                print(f"Not enough bubbles detected: {len(bubble_centers)}")
                return None
            
            # Convert to numpy array
            centers = np.array(bubble_centers)
            
            # Use K-means clustering to find columns
            from sklearn.cluster import KMeans
            
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
                return None
            
            # Build grid structure
            grid = {}
            question_num = 1
            
            for col_idx, col_bubbles in enumerate(column_clusters):
                print(f"Column {col_idx + 1}: {len(col_bubbles)} bubbles")
                
                # Group bubbles by rows (questions) - 4 bubbles per question
                questions_in_column = []
                bubbles_per_question = 4
                
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
            
            print(f"Built grid with {len(grid)} questions")
            return grid
            
        except Exception as e:
            print(f"Error in grid building: {str(e)}")
            return None
    
    def detect_filled(self, image: np.ndarray, grid: Dict[str, Any], threshold: float) -> Dict[str, str]:
        """
        Detect which bubbles are filled based on darkness threshold
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            filled_bubbles = {}
            
            for question_num, options in grid.items():
                option_scores = {}
                
                for option, (x, y) in options.items():
                    # Extract bubble region with adaptive size
                    # Calculate bubble size based on image dimensions
                    image_area = gray.shape[0] * gray.shape[1]
                    bubble_radius = max(8, int(np.sqrt(image_area) / 50))  # Adaptive bubble size
                    
                    x1, y1 = max(0, x - bubble_radius), max(0, y - bubble_radius)
                    x2, y2 = min(gray.shape[1], x + bubble_radius), min(gray.shape[0], y + bubble_radius)
                    
                    bubble_region = gray[y1:y2, x1:x2]
                    
                    if bubble_region.size > 0:
                        # Calculate multiple metrics
                        mean_intensity = np.mean(bubble_region)
                        min_intensity = np.min(bubble_region)
                        std_intensity = np.std(bubble_region)
                        
                        # Combined score (lower is more likely filled)
                        # Weight mean intensity heavily, but also consider min and std
                        combined_score = mean_intensity * 0.7 + min_intensity * 0.2 + std_intensity * 0.1
                        option_scores[option] = combined_score
                
                # Find the darkest option (most likely filled)
                if option_scores:
                    darkest_option = min(option_scores, key=option_scores.get)
                    darkest_score = option_scores[darkest_option]
                    
                    # Check if it's significantly darker than other options
                    other_scores = [score for opt, score in option_scores.items() if opt != darkest_option]
                    if other_scores:
                        avg_other_score = np.mean(other_scores)
                        std_other_score = np.std(other_scores)
                        
                        # More sophisticated thresholding
                        # Check if darkest is significantly below average
                        if darkest_score < avg_other_score - max(15, std_other_score * 0.5):
                            filled_bubbles[question_num] = darkest_option
                        else:
                            filled_bubbles[question_num] = None  # No clear selection
                    else:
                        filled_bubbles[question_num] = None
                else:
                    filled_bubbles[question_num] = None
            
            print(f"Detected filled bubbles for {len([k for k, v in filled_bubbles.items() if v is not None])} questions")
            return filled_bubbles
            
        except Exception as e:
            print(f"Error in filled detection: {str(e)}")
            return {}
    
    def evaluate(self, filled_bubbles: Dict[str, str], answer_key: Dict[str, str], 
                subjects: Dict[str, str]) -> Dict[str, Any]:
        """
        Evaluate answers and calculate scores
        """
        try:
            # Initialize results
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
            
        except Exception as e:
            print(f"Error in evaluation: {str(e)}")
            return {}
    
    def create_annotated_image(self, image: np.ndarray, grid: Dict[str, Any], 
                             filled_bubbles: Dict[str, str], results: Dict[str, Any]) -> np.ndarray:
        """
        Create annotated image showing detected bubbles and results
        """
        try:
            # Create a copy of the image
            annotated = image.copy()
            
            # Define colors
            correct_color = (0, 255, 0)  # Green for correct
            incorrect_color = (0, 0, 255)  # Red for incorrect
            invalid_color = (0, 255, 255)  # Yellow for invalid
            unanswered_color = (128, 128, 128)  # Gray for unanswered
            
            for question_num, options in grid.items():
                detected = filled_bubbles.get(question_num)
                correct_answer = results['per_question'].get(question_num, {})
                
                for option, (x, y) in options.items():
                    # Draw bubble circle
                    cv2.circle(annotated, (x, y), self.bubble_diameter // 2, (255, 255, 255), 2)
                    
                    # Color based on result
                    if detected == option:
                        if correct_answer.get('correct', False):
                            color = correct_color
                        elif not correct_answer.get('valid', True):
                            color = invalid_color
                        else:
                            color = incorrect_color
                    elif option == results['per_question'].get(question_num, {}).get('detected', ''):
                        color = unanswered_color
                    else:
                        continue
                    
                    # Fill the bubble
                    cv2.circle(annotated, (x, y), self.bubble_diameter // 2 - 2, color, -1)
                    
                    # Add option label
                    cv2.putText(annotated, option, (x - 5, y + 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            return annotated
            
        except Exception as e:
            print(f"Error in annotation: {str(e)}")
            return image
