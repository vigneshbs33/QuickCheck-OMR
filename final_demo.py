"""
Final Demo of QuickCheck OMR Evaluation System
Demonstrates the complete system with all features
Developed by InteliCat Team
"""

import cv2
import json
import os
from pathlib import Path
import streamlit as st
from omr.optimized_engine import OptimizedOMRProcessor


def demo_quickcheck_system():
    """Demonstrate the complete QuickCheck system"""
    print("🎯 QuickCheck OMR Evaluation System - Final Demo")
    print("=" * 60)
    
    # Initialize the optimized processor
    processor = OptimizedOMRProcessor()
    
    # Load the correct answer key
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
    
    # Test on a few representative images
    sample_images = [
        "sample-to-check/set_a/Img1.jpeg",
        "sample-to-check/set_a/Img2.jpeg", 
        "sample-to-check/set_a/Img3.jpeg"
    ]
    
    print(f"\n🧪 Testing on {len(sample_images)} representative images...")
    
    results = []
    
    for i, image_path in enumerate(sample_images):
        print(f"\n📸 Processing {Path(image_path).name} ({i+1}/{len(sample_images)})")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Failed to load {image_path}")
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
                    'file': Path(image_path).name,
                    'accuracy': accuracy,
                    'detected_questions': detected_questions,
                    'total_score': total_score,
                    'grade': grade,
                    'result': result
                })
                
                print(f"   ✅ Success: {accuracy:.1f}% accuracy, {detected_questions} questions detected, Grade: {grade}")
                
                # Display subject-wise scores
                print(f"   📊 Subject-wise scores:")
                for subject, data in result['subjects'].items():
                    subject_accuracy = (data['score'] / data['out_of']) * 100
                    print(f"      {subject}: {data['score']}/{data['out_of']} ({subject_accuracy:.1f}%)")
                
                # Save annotated image
                if 'annotated_image' in result:
                    output_path = f"sample_data/demo_annotated_{Path(image_path).name}"
                    cv2.imwrite(output_path, result['annotated_image'])
                    print(f"   💾 Annotated image saved: {output_path}")
                
            else:
                print(f"   ❌ Failed to process {Path(image_path).name}")
                
        except Exception as e:
            print(f"   ❌ Error processing {Path(image_path).name}: {str(e)}")
    
    # Summary
    if results:
        print(f"\n📊 DEMO RESULTS SUMMARY")
        print("=" * 60)
        
        accuracies = [r['accuracy'] for r in results]
        detected_counts = [r['detected_questions'] for r in results]
        total_scores = [r['total_score'] for r in results]
        
        print(f"✅ Successfully processed: {len(results)}/{len(sample_images)} images")
        print(f"📈 Average Accuracy: {sum(accuracies)/len(accuracies):.1f}%")
        print(f"🏆 Best Accuracy: {max(accuracies):.1f}%")
        print(f"📊 Average Questions Detected: {sum(detected_counts)/len(detected_counts):.1f}/100")
        print(f"🎯 Average Score: {sum(total_scores)/len(total_scores):.1f}/100")
        
        # Save demo results
        with open('sample_data/demo_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Demo results saved to: sample_data/demo_results.json")
        
        print(f"\n🎉 QuickCheck System Demo Complete!")
        print(f"🚀 The system is ready for production use!")
        
    else:
        print("❌ No images were successfully processed in the demo!")


def show_system_capabilities():
    """Show the system's capabilities"""
    print(f"\n🔧 QuickCheck System Capabilities:")
    print("=" * 60)
    print("✅ Template-free OMR processing")
    print("✅ Automatic orientation correction")
    print("✅ Robust bubble detection using multiple algorithms")
    print("✅ Adaptive position calibration")
    print("✅ Subject-wise scoring and analysis")
    print("✅ Comprehensive result reporting")
    print("✅ Annotated image generation")
    print("✅ JSON and CSV export functionality")
    print("✅ Batch processing support")
    print("✅ Web-based user interface")
    print("✅ High accuracy on your specific OMR format")


def show_usage_instructions():
    """Show usage instructions"""
    print(f"\n📖 How to Use QuickCheck:")
    print("=" * 60)
    print("1. 🚀 Start the web interface:")
    print("   streamlit run app.py")
    print()
    print("2. 📸 Upload OMR sheet images (JPEG/PNG)")
    print("3. 📝 Enter answer key in JSON format:")
    print('   {"1": "A", "2": "B", "3": "C", ...}')
    print("4. 📚 Specify subject ranges:")
    print("   1-20:Python,21-40:EDA,41-60:MySQL,61-80:PowerBI,81-100:AdvStats")
    print("5. ⚙️ Adjust processing parameters if needed")
    print("6. 🎯 Click 'Evaluate' to process the OMR sheet")
    print("7. 📊 View results and download reports")
    print()
    print("🔧 For batch processing:")
    print("   - Place images in a folder")
    print("   - Enter folder path in the UI")
    print("   - Click 'Process Batch Folder'")


def main():
    """Main demo function"""
    show_system_capabilities()
    demo_quickcheck_system()
    show_usage_instructions()
    
    print(f"\n🎯 QuickCheck - Making OMR evaluation simple and accurate!")
    print(f"💼 Developed by InteliCat Team")


if __name__ == "__main__":
    main()
