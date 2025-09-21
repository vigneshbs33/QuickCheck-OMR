"""
Test script for QuickCheck OMR Evaluator
Tests the system with sample data from sample-to-check directory
"""

import cv2
import json
import os
from pathlib import Path
from omr.adaptive_engine import AdaptiveOMRProcessor

def test_with_sample_data():
    """Test the OMR processor with sample data"""
    
    # Initialize processor
    processor = AdaptiveOMRProcessor()
    
    # Load answer key
    with open('sample_data/sample_answer_key.json', 'r') as f:
        answer_key_data = json.load(f)
    
    # Convert to simple format
    answer_key = {}
    subjects = {}
    
    for subject_name, subject_data in answer_key_data['subjects'].items():
        for q_num, answer in subject_data['questions'].items():
            answer_key[q_num] = answer
            subjects[q_num] = subject_name
    
    print("Answer key loaded successfully!")
    print(f"Total questions: {len(answer_key)}")
    print(f"Subjects: {list(set(subjects.values()))}")
    
    # Test with sample images
    sample_images = [
        'sample_data/test_image.jpeg',
        'sample_data/test_image2.jpeg', 
        'sample_data/test_image3.jpeg'
    ]
    
    results = []
    
    for img_path in sample_images:
        if os.path.exists(img_path):
            print(f"\nProcessing: {img_path}")
            
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to load image: {img_path}")
                continue
            
            print(f"Image shape: {image.shape}")
            
            # Process OMR
            result = processor.process_omr(image, answer_key, subjects, bubble_threshold=0.4)
            
            if result:
                print(f"✅ Successfully processed!")
                print(f"Total Score: {result['total_score']}/100")
                print(f"Percentage: {result['percentage']:.1f}%")
                print(f"Grade: {result['grade']}")
                
                # Subject-wise scores
                print("\nSubject-wise scores:")
                for subject, data in result['subjects'].items():
                    print(f"  {subject}: {data['score']}/{data['out_of']} ({(data['score']/data['out_of']*100):.1f}%)")
                
                # Save annotated image
                if 'annotated_image' in result:
                    output_path = f"sample_data/annotated_{Path(img_path).name}"
                    cv2.imwrite(output_path, result['annotated_image'])
                    print(f"Annotated image saved: {output_path}")
                
                results.append({
                    'image': img_path,
                    'result': result
                })
            else:
                print(f"❌ Failed to process image: {img_path}")
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Total images processed: {len(results)}")
    
    if results:
        avg_score = sum(r['result']['total_score'] for r in results) / len(results)
        avg_percentage = sum(r['result']['percentage'] for r in results) / len(results)
        print(f"Average score: {avg_score:.1f}/100")
        print(f"Average percentage: {avg_percentage:.1f}%")
        
        # Save results
        with open('sample_data/test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print("Results saved to: sample_data/test_results.json")

if __name__ == "__main__":
    test_with_sample_data()
