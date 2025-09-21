"""
Comprehensive analysis of sample data to create the most accurate OMR model
Developed by InteliCat Team
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import statistics

# Import our engines
from omr.adaptive_engine import AdaptiveOMRProcessor
from omr.accurate_engine import AccurateOMRProcessor
from omr.template_engine import TemplateOMRProcessor
from omr.calibrated_engine import CalibratedOMRProcessor


class SampleDataAnalyzer:
    """Analyzes sample data to optimize OMR model accuracy"""
    
    def __init__(self):
        self.sample_dir = Path("sample-to-check/set_a")
        self.answer_key_path = Path("sample-to-check/set_a.json")
        self.results = {}
        
        # Load answer key
        with open(self.answer_key_path, 'r') as f:
            self.answer_key_data = json.load(f)
        
        # Convert to simple format
        self.answer_key = {}
        self.subjects = {}
        
        for subject_name, subject_data in self.answer_key_data['subjects'].items():
            for q_num, answer in subject_data['questions'].items():
                self.answer_key[q_num] = answer
                self.subjects[q_num] = subject_name
    
    def analyze_all_samples(self):
        """Analyze all sample images and compare different engines"""
        print("🔍 Analyzing all sample data...")
        
        # Get all image files
        image_files = list(self.sample_dir.glob("*.jpeg"))
        print(f"Found {len(image_files)} sample images")
        
        # Test different engines
        engines = {
            "Adaptive": AdaptiveOMRProcessor(),
            "Accurate": AccurateOMRProcessor(),
            "Template": TemplateOMRProcessor(),
            "Calibrated": CalibratedOMRProcessor()
        }
        
        engine_results = defaultdict(list)
        
        for i, image_file in enumerate(image_files[:10]):  # Test first 10 images
            print(f"\n📸 Processing {image_file.name} ({i+1}/10)")
            
            # Load image
            image = cv2.imread(str(image_file))
            if image is None:
                print(f"❌ Failed to load {image_file.name}")
                continue
            
            print(f"   Image shape: {image.shape}")
            
            # Test each engine
            for engine_name, processor in engines.items():
                try:
                    result = processor.process_omr(image, self.answer_key, self.subjects)
                    
                    if result:
                        accuracy = result['percentage']
                        detected_questions = len([k for k, v in result['per_question'].items() if v['detected'] != 'None'])
                        
                        engine_results[engine_name].append({
                            'file': image_file.name,
                            'accuracy': accuracy,
                            'detected_questions': detected_questions,
                            'total_score': result['total_score'],
                            'result': result
                        })
                        
                        print(f"   {engine_name}: {accuracy:.1f}% accuracy, {detected_questions} questions detected")
                    else:
                        print(f"   {engine_name}: Failed to process")
                        
                except Exception as e:
                    print(f"   {engine_name}: Error - {str(e)}")
        
        # Analyze results
        self.analyze_engine_performance(engine_results)
        
        # Find the best performing image for calibration
        self.find_best_calibration_image(engine_results)
        
        return engine_results
    
    def analyze_engine_performance(self, engine_results):
        """Analyze performance of different engines"""
        print("\n📊 Engine Performance Analysis:")
        print("=" * 50)
        
        for engine_name, results in engine_results.items():
            if not results:
                print(f"{engine_name}: No successful results")
                continue
                
            accuracies = [r['accuracy'] for r in results]
            detected_counts = [r['detected_questions'] for r in results]
            
            print(f"\n{engine_name}:")
            print(f"  Average Accuracy: {statistics.mean(accuracies):.1f}%")
            print(f"  Best Accuracy: {max(accuracies):.1f}%")
            print(f"  Worst Accuracy: {min(accuracies):.1f}%")
            print(f"  Average Questions Detected: {statistics.mean(detected_counts):.1f}/100")
            print(f"  Success Rate: {len(results)}/10 images")
    
    def find_best_calibration_image(self, engine_results):
        """Find the best image for calibration"""
        print("\n🎯 Finding best calibration image...")
        
        best_image = None
        best_score = 0
        
        for engine_name, results in engine_results.items():
            for result in results:
                # Score based on both accuracy and detection rate
                detection_rate = result['detected_questions'] / 100
                combined_score = result['accuracy'] * 0.7 + detection_rate * 100 * 0.3
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_image = result['file']
        
        if best_image:
            print(f"Best calibration image: {best_image} (score: {best_score:.1f})")
            return best_image
        else:
            print("No suitable calibration image found")
            return None
    
    def create_optimized_engine(self, best_image_path):
        """Create an optimized engine based on the best performing image"""
        print(f"\n🚀 Creating optimized engine based on {best_image_path}...")
        
        # Load the best image
        image = cv2.imread(str(self.sample_dir / best_image_path))
        if image is None:
            print("Failed to load best image")
            return None
        
        # Create a learning-based processor
        return LearningOMRProcessor(image, self.answer_key, self.subjects)
    
    def run_comprehensive_test(self):
        """Run comprehensive test on all sample data"""
        print("🧪 Running comprehensive test...")
        
        # Analyze all samples
        engine_results = self.analyze_all_samples()
        
        # Find best calibration image
        best_image = self.find_best_calibration_image(engine_results)
        
        if best_image:
            # Create optimized engine
            optimized_processor = self.create_optimized_engine(best_image)
            
            if optimized_processor:
                # Test optimized engine on all samples
                self.test_optimized_engine(optimized_processor, engine_results)
        
        return engine_results
    
    def test_optimized_engine(self, optimized_processor, engine_results):
        """Test the optimized engine on all samples"""
        print(f"\n🧪 Testing optimized engine on all samples...")
        
        # Get all image files
        image_files = list(self.sample_dir.glob("*.jpeg"))
        optimized_results = []
        
        for i, image_file in enumerate(image_files[:10]):  # Test first 10 images
            print(f"\n📸 Testing optimized engine on {image_file.name} ({i+1}/10)")
            
            # Load image
            image = cv2.imread(str(image_file))
            if image is None:
                continue
            
            try:
                result = optimized_processor.process_omr(image, self.answer_key, self.subjects)
                
                if result:
                    accuracy = result['percentage']
                    detected_questions = len([k for k, v in result['per_question'].items() if v['detected'] != 'None'])
                    
                    optimized_results.append({
                        'file': image_file.name,
                        'accuracy': accuracy,
                        'detected_questions': detected_questions,
                        'total_score': result['total_score'],
                        'result': result
                    })
                    
                    print(f"   Optimized: {accuracy:.1f}% accuracy, {detected_questions} questions detected")
                else:
                    print(f"   Optimized: Failed to process")
                    
            except Exception as e:
                print(f"   Optimized: Error - {str(e)}")
        
        # Compare with other engines
        if optimized_results:
            accuracies = [r['accuracy'] for r in optimized_results]
            detected_counts = [r['detected_questions'] for r in optimized_results]
            
            print(f"\n📊 Optimized Engine Performance:")
            print(f"  Average Accuracy: {statistics.mean(accuracies):.1f}%")
            print(f"  Best Accuracy: {max(accuracies):.1f}%")
            print(f"  Average Questions Detected: {statistics.mean(detected_counts):.1f}/100")
            print(f"  Success Rate: {len(optimized_results)}/10 images")
            
            # Compare with best existing engine
            print(f"\n🆚 Comparison with Calibrated Engine:")
            calibrated_results = engine_results.get('Calibrated', [])
            if calibrated_results:
                calibrated_accuracies = [r['accuracy'] for r in calibrated_results]
                print(f"  Calibrated Average: {statistics.mean(calibrated_accuracies):.1f}%")
                print(f"  Optimized Average: {statistics.mean(accuracies):.1f}%")
                
                improvement = statistics.mean(accuracies) - statistics.mean(calibrated_accuracies)
                if improvement > 0:
                    print(f"  🎉 Improvement: +{improvement:.1f}%")
                else:
                    print(f"  📉 Change: {improvement:.1f}%")


class LearningOMRProcessor:
    """Learning-based OMR processor that adapts to the specific format"""
    
    def __init__(self, reference_image, answer_key, subjects):
        self.answer_key = answer_key
        self.subjects = subjects
        self.reference_image = reference_image
        
        # Learn from reference image
        self.bubble_positions = self.learn_bubble_positions(reference_image)
        self.bubble_size = self.learn_bubble_size(reference_image)
        self.threshold_params = self.learn_threshold_params(reference_image)
        
        print(f"Learned {len(self.bubble_positions)} bubble positions")
        print(f"Learned bubble size: {self.bubble_size}")
    
    def learn_bubble_positions(self, image):
        """Learn bubble positions from reference image"""
        # This would implement advanced learning algorithms
        # For now, use the adaptive approach
        processor = AdaptiveOMRProcessor()
        return processor.scale_positions_to_image(image)
    
    def learn_bubble_size(self, image):
        """Learn optimal bubble size from reference image"""
        # Analyze image to determine optimal bubble size
        height, width = image.shape[:2]
        return max(8, int(np.sqrt(height * width) / 60))
    
    def learn_threshold_params(self, image):
        """Learn optimal threshold parameters from reference image"""
        # Analyze image characteristics to determine optimal thresholds
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        return {
            'base_threshold': mean_intensity - std_intensity * 0.5,
            'adaptive_factor': 0.3
        }
    
    def process_omr(self, image, answer_key, subjects, bubble_threshold=0.4):
        """Process OMR using learned parameters"""
        try:
            # Use learned parameters for processing
            filled_bubbles = self.detect_filled_bubbles_learned(image)
            results = self.evaluate_answers(filled_bubbles, answer_key, subjects)
            
            # Create annotated image
            annotated_image = self.create_annotated_image(image, filled_bubbles, results)
            results['annotated_image'] = annotated_image
            
            return results
            
        except Exception as e:
            print(f"Error in learned processing: {str(e)}")
            return None
    
    def detect_filled_bubbles_learned(self, image):
        """Detect filled bubbles using learned parameters"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        filled_bubbles = {}
        
        for question_num, options in self.bubble_positions.items():
            option_scores = {}
            
            for option, (x, y) in options.items():
                # Use learned bubble size
                bubble_radius = self.bubble_size
                x1, y1 = max(0, x - bubble_radius), max(0, y - bubble_radius)
                x2, y2 = min(gray.shape[1], x + bubble_radius), min(gray.shape[0], y + bubble_radius)
                
                bubble_region = gray[y1:y2, x1:x2]
                
                if bubble_region.size > 0:
                    # Use learned threshold parameters
                    mean_intensity = np.mean(bubble_region)
                    option_scores[option] = mean_intensity
            
            # Apply learned thresholding
            if option_scores:
                darkest_option = min(option_scores, key=option_scores.get)
                darkest_score = option_scores[darkest_option]
                
                other_scores = [score for opt, score in option_scores.items() if opt != darkest_option]
                if other_scores:
                    threshold_value = self.threshold_params['base_threshold']
                    
                    if darkest_score < threshold_value:
                        filled_bubbles[question_num] = darkest_option
                    else:
                        filled_bubbles[question_num] = None
                else:
                    filled_bubbles[question_num] = None
            else:
                filled_bubbles[question_num] = None
        
        return filled_bubbles
    
    def evaluate_answers(self, filled_bubbles, answer_key, subjects):
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
        for question_num in range(1, 101):
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
        results['percentage'] = (total_correct / 100) * 100
        
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
    
    def create_annotated_image(self, image, filled_bubbles, results):
        """Create annotated image showing results"""
        if len(image.shape) == 2:
            annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            annotated = image.copy()
        
        # Define colors
        correct_color = (0, 255, 0)  # Green
        incorrect_color = (0, 0, 255)  # Red
        invalid_color = (0, 255, 255)  # Yellow
        
        for question_num, options in self.bubble_positions.items():
            detected = filled_bubbles.get(question_num)
            question_result = results['per_question'].get(question_num, {})
            
            for option, (x, y) in options.items():
                # Draw bubble circle
                cv2.circle(annotated, (x, y), self.bubble_size, (255, 255, 255), 2)
                
                # Color based on result
                if detected == option:
                    if question_result.get('correct', False):
                        color = correct_color
                    elif not question_result.get('valid', True):
                        color = invalid_color
                    else:
                        color = incorrect_color
                    
                    # Fill the bubble
                    cv2.circle(annotated, (x, y), self.bubble_size - 2, color, -1)
                    
                    # Add option label
                    cv2.putText(annotated, option, (x - 5, y + 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated


def main():
    """Main analysis function"""
    print("🎯 QuickCheck Sample Data Analysis")
    print("=" * 50)
    
    analyzer = SampleDataAnalyzer()
    results = analyzer.run_comprehensive_test()
    
    print("\n✅ Analysis complete!")
    print("Check the results above to see which engine performs best on your data.")


if __name__ == "__main__":
    main()
