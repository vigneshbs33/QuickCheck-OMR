"""
Unit tests for QuickCheck OMR Engine
Developed by InteliCat Team
"""

import unittest
import cv2
import numpy as np
import json
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from omr.adaptive_engine import AdaptiveOMRProcessor


class TestOMREngine(unittest.TestCase):
    """Test cases for OMR processing engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = AdaptiveOMRProcessor()
        
        # Load test answer key
        answer_key_path = Path(__file__).parent.parent / "sample_data" / "sample_answer_key.json"
        with open(answer_key_path, 'r') as f:
            answer_key_data = json.load(f)
        
        # Convert to simple format
        self.answer_key = {}
        self.subjects = {}
        
        for subject_name, subject_data in answer_key_data['subjects'].items():
            for q_num, answer in subject_data['questions'].items():
                self.answer_key[q_num] = answer
                self.subjects[q_num] = subject_name
    
    def test_processor_initialization(self):
        """Test that processor initializes correctly"""
        self.assertIsNotNone(self.processor)
        self.assertEqual(self.processor.total_questions, 100)
        self.assertEqual(self.processor.num_columns, 5)
        self.assertEqual(self.processor.questions_per_column, 20)
    
    def test_reference_positions_building(self):
        """Test that reference positions are built correctly"""
        positions = self.processor.ref_positions
        self.assertIsInstance(positions, dict)
        self.assertEqual(len(positions), 100)  # Should have 100 questions
        
        # Check that each question has 4 options
        for q_num, options in positions.items():
            self.assertEqual(len(options), 4)
            self.assertIn('A', options)
            self.assertIn('B', options)
            self.assertIn('C', options)
            self.assertIn('D', options)
    
    def test_position_scaling(self):
        """Test position scaling to different image dimensions"""
        # Create a test image
        test_image = np.zeros((500, 600, 3), dtype=np.uint8)
        
        # Scale positions
        scaled_positions = self.processor.scale_positions_to_image(test_image)
        
        self.assertIsInstance(scaled_positions, dict)
        self.assertEqual(len(scaled_positions), 100)
        
        # Check that positions are within image bounds
        for q_num, options in scaled_positions.items():
            for option, (x, y) in options.items():
                self.assertGreaterEqual(x, 0)
                self.assertLess(x, test_image.shape[1])
                self.assertGreaterEqual(y, 0)
                self.assertLess(y, test_image.shape[0])
    
    def test_answer_evaluation(self):
        """Test answer evaluation logic"""
        # Test data
        filled_bubbles = {
            "1": "A",  # Correct
            "2": "B",  # Incorrect
            "3": "C",  # Correct
            "4": None,  # Unanswered
        }
        
        # Mock answer key
        answer_key = {
            "1": "A",
            "2": "A",  # Student answered B, correct is A
            "3": "C",
            "4": "D",  # Student didn't answer, correct is D
        }
        
        subjects = {
            "1": "Python",
            "2": "Python", 
            "3": "Python",
            "4": "Python",
        }
        
        # Evaluate
        results = self.processor.evaluate_answers(filled_bubbles, answer_key, subjects)
        
        # Check results
        self.assertEqual(results['total_score'], 2)  # 2 correct out of 4
        self.assertEqual(results['percentage'], 50.0)
        self.assertEqual(results['grade'], 'C')
        
        # Check per-question results
        self.assertTrue(results['per_question']['1']['correct'])
        self.assertFalse(results['per_question']['2']['correct'])
        self.assertTrue(results['per_question']['3']['correct'])
        self.assertFalse(results['per_question']['4']['correct'])
    
    def test_grade_calculation(self):
        """Test grade calculation logic"""
        # Test different score ranges
        test_cases = [
            (95, "A"),  # >= 85%
            (80, "B"),  # 70-84%
            (60, "C"),  # 50-69%
            (40, "D"),  # < 50%
        ]
        
        for percentage, expected_grade in test_cases:
            # Create mock results
            filled_bubbles = {str(i): "A" for i in range(1, 101)}
            answer_key = {str(i): "A" if i <= percentage else "B" for i in range(1, 101)}
            subjects = {str(i): "Test" for i in range(1, 101)}
            
            results = self.processor.evaluate_answers(filled_bubbles, answer_key, subjects)
            self.assertEqual(results['grade'], expected_grade)
    
    def test_image_processing_pipeline(self):
        """Test the complete image processing pipeline"""
        # Create a mock OMR image
        mock_image = np.ones((1122, 1200, 3), dtype=np.uint8) * 255  # White background
        
        # Add some mock bubbles (dark circles)
        for i in range(10):
            x = 100 + i * 50
            y = 100 + i * 30
            cv2.circle(mock_image, (x, y), 10, (0, 0, 0), -1)  # Black filled circles
        
        # Process the image
        result = self.processor.process_omr(mock_image, self.answer_key, self.subjects)
        
        # Should return a result (even if not perfect)
        self.assertIsNotNone(result)
        self.assertIn('total_score', result)
        self.assertIn('percentage', result)
        self.assertIn('grade', result)
        self.assertIn('subjects', result)
        self.assertIn('per_question', result)
    
    def test_batch_processing(self):
        """Test batch processing functionality"""
        # This would test batch processing if implemented
        # For now, just test that the processor can handle multiple images
        mock_image = np.ones((1122, 1200, 3), dtype=np.uint8) * 255
        
        # Process multiple times
        for i in range(3):
            result = self.processor.process_omr(mock_image, self.answer_key, self.subjects)
            self.assertIsNotNone(result)


if __name__ == '__main__':
    # Run tests
    unittest.main()
