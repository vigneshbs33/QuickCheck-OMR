"""
Comprehensive test of the optimized OMR engine on all sample data
Developed by InteliCat Team
"""

import cv2
import json
import os
from pathlib import Path
import statistics
from omr.optimized_engine import OptimizedOMRProcessor


def test_optimized_engine():
    """Test the optimized engine on all sample data"""
    print("🚀 Testing Optimized OMR Engine on All Sample Data")
    print("=" * 60)
    
    # Initialize processor
    processor = OptimizedOMRProcessor()
    
    # Load answer key
    answer_key_path = Path("sample-to-check/set_a.json")
    with open(answer_key_path, 'r') as f:
        answer_key_data = json.load(f)
    
    # Convert to simple format
    answer_key = {}
    subjects = {}
    
    for subject_name, subject_data in answer_key_data['subjects'].items():
        for q_num, answer in subject_data['questions'].items():
            answer_key[q_num] = answer
            subjects[q_num] = subject_name
    
    print(f"✅ Loaded answer key for {len(answer_key)} questions")
    print(f"📚 Subjects: {list(set(subjects.values()))}")
    
    # Get all sample images
    sample_dir = Path("sample-to-check/set_a")
    image_files = list(sample_dir.glob("*.jpeg"))
    print(f"📸 Found {len(image_files)} sample images")
    
    # Test on all images
    results = []
    successful_processing = 0
    
    for i, image_file in enumerate(image_files):
        print(f"\n📸 Processing {image_file.name} ({i+1}/{len(image_files)})")
        
        # Load image
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"❌ Failed to load {image_file.name}")
            continue
        
        print(f"   Image shape: {image.shape}")
        
        try:
            # Process with optimized engine
            result = processor.process_omr(image, answer_key, subjects, bubble_threshold=0.4)
            
            if result:
                accuracy = result['percentage']
                detected_questions = len([k for k, v in result['per_question'].items() if v['detected'] != 'None'])
                total_score = result['total_score']
                grade = result['grade']
                
                results.append({
                    'file': image_file.name,
                    'accuracy': accuracy,
                    'detected_questions': detected_questions,
                    'total_score': total_score,
                    'grade': grade,
                    'result': result
                })
                
                successful_processing += 1
                print(f"   ✅ Success: {accuracy:.1f}% accuracy, {detected_questions} questions detected, Grade: {grade}")
                
                # Save annotated image
                if 'annotated_image' in result:
                    output_path = f"sample_data/optimized_annotated_{image_file.name}"
                    cv2.imwrite(output_path, result['annotated_image'])
                    print(f"   💾 Annotated image saved: {output_path}")
                
            else:
                print(f"   ❌ Failed to process {image_file.name}")
                
        except Exception as e:
            print(f"   ❌ Error processing {image_file.name}: {str(e)}")
    
    # Analyze results
    if results:
        print(f"\n📊 OPTIMIZED ENGINE PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        accuracies = [r['accuracy'] for r in results]
        detected_counts = [r['detected_questions'] for r in results]
        total_scores = [r['total_score'] for r in results]
        
        print(f"✅ Successfully processed: {successful_processing}/{len(image_files)} images")
        print(f"📈 Average Accuracy: {statistics.mean(accuracies):.1f}%")
        print(f"🏆 Best Accuracy: {max(accuracies):.1f}%")
        print(f"📉 Worst Accuracy: {min(accuracies):.1f}%")
        print(f"📊 Average Questions Detected: {statistics.mean(detected_counts):.1f}/100")
        print(f"🎯 Average Score: {statistics.mean(total_scores):.1f}/100")
        
        # Grade distribution
        grades = [r['grade'] for r in results]
        grade_counts = {}
        for grade in grades:
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
        
        print(f"\n📚 Grade Distribution:")
        for grade, count in sorted(grade_counts.items()):
            print(f"   Grade {grade}: {count} images")
        
        # Subject-wise analysis
        print(f"\n📖 Subject-wise Performance:")
        subject_totals = {}
        subject_corrects = {}
        
        for result in results:
            for subject, data in result['result']['subjects'].items():
                if subject not in subject_totals:
                    subject_totals[subject] = 0
                    subject_corrects[subject] = 0
                subject_totals[subject] += data['out_of']
                subject_corrects[subject] += data['score']
        
        for subject in subject_totals:
            avg_accuracy = (subject_corrects[subject] / subject_totals[subject]) * 100
            print(f"   {subject}: {avg_accuracy:.1f}% average accuracy")
        
        # Save detailed results
        with open('sample_data/optimized_test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: sample_data/optimized_test_results.json")
        
        # Find best performing images
        best_images = sorted(results, key=lambda x: x['accuracy'], reverse=True)[:5]
        print(f"\n🏆 Top 5 Best Performing Images:")
        for i, img in enumerate(best_images, 1):
            print(f"   {i}. {img['file']}: {img['accuracy']:.1f}% accuracy, Grade {img['grade']}")
        
        # Find images with highest detection rates
        best_detection = sorted(results, key=lambda x: x['detected_questions'], reverse=True)[:5]
        print(f"\n🎯 Top 5 Images with Highest Detection Rates:")
        for i, img in enumerate(best_detection, 1):
            print(f"   {i}. {img['file']}: {img['detected_questions']}/100 questions detected")
        
    else:
        print("❌ No images were successfully processed!")
    
    print(f"\n✅ Testing complete!")


if __name__ == "__main__":
    test_optimized_engine()
