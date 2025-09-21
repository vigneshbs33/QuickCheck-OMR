"""
QuickCheck - OMR Evaluation App
Developed by InteliCat Team

This implementation is provided to InteliCat for exclusive use under the project name QuickCheck.
"""

import streamlit as st
import cv2
import numpy as np
import json
import pandas as pd
from PIL import Image
import io
import os
from pathlib import Path

# Import our OMR engine
from omr.optimized_engine import OptimizedOMRProcessor

# Page configuration
st.set_page_config(
    page_title="QuickCheck - OMR Evaluator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for branding
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .intelicat-brand {
        color: #ff6b6b;
        font-weight: bold;
    }
    .result-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .score-display {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    .grade-A { color: #28a745; }
    .grade-B { color: #17a2b8; }
    .grade-C { color: #ffc107; }
    .grade-D { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">QuickCheck</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">OMR Evaluation System by <span class="intelicat-brand">InteliCat</span></p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'omr_processor' not in st.session_state:
        st.session_state.omr_processor = OptimizedOMRProcessor()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Answer key input
        st.subheader("Answer Key")
        answer_key_format = st.radio(
            "Answer Key Format",
            ["100-letter string", "JSON format"],
            help="Choose how to input the answer key"
        )
        
        if answer_key_format == "100-letter string":
            answer_key_input = st.text_area(
                "Enter 100-letter answer key (e.g., ABCD...)",
                value="A" * 100,
                height=100,
                help="Enter exactly 100 letters representing answers A, B, C, or D"
            )
        else:
            answer_key_input = st.text_area(
                "Enter JSON answer key",
                value=json.dumps({"1": "A", "2": "B", "3": "C"}, indent=2),
                height=200,
                help="Enter JSON format: {\"1\": \"A\", \"2\": \"B\", ...}"
            )
        
        # Subject ranges
        st.subheader("Subject Ranges")
        subject_ranges = st.text_area(
            "Subject ranges (e.g., 1-20:Python,21-40:EDA,41-60:MySQL,61-80:PowerBI,81-100:AdvStats)",
            value="1-20:Python,21-40:EDA,41-60:MySQL,61-80:PowerBI,81-100:AdvStats",
            help="Format: start-end:SubjectName,start-end:SubjectName,..."
        )
        
        # Processing parameters
        st.subheader("Processing Parameters")
        bubble_threshold = st.slider(
            "Bubble Selection Threshold",
            min_value=0.1,
            max_value=0.8,
            value=0.4,
            step=0.05,
            help="Threshold for detecting filled bubbles (0.4 = 40% of bubble area must be dark)"
        )
        
        # Grade thresholds
        st.subheader("Grade Thresholds")
        grade_a = st.number_input("Grade A threshold (%)", min_value=0, max_value=100, value=85)
        grade_b = st.number_input("Grade B threshold (%)", min_value=0, max_value=100, value=70)
        grade_c = st.number_input("Grade C threshold (%)", min_value=0, max_value=100, value=50)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Upload OMR Sheet")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an OMR sheet image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a photo of the OMR sheet. The app will auto-correct orientation and detect bubbles."
        )
        
        # Batch processing option
        st.subheader("📁 Batch Processing")
        batch_folder = st.text_input(
            "Batch folder path (optional)",
            placeholder="e.g., sample_data/",
            help="Process all images in a folder"
        )
        
        if st.button("Process Batch Folder", disabled=not batch_folder):
            if os.path.exists(batch_folder):
                process_batch_folder(batch_folder, answer_key_input, answer_key_format, subject_ranges, bubble_threshold, grade_a, grade_b, grade_c)
            else:
                st.error("Folder does not exist!")
    
    with col2:
        st.header("📊 Results")
        
        if uploaded_file is not None:
            # Process single image
            process_single_image(uploaded_file, answer_key_input, answer_key_format, subject_ranges, bubble_threshold, grade_a, grade_b, grade_c)
        else:
            st.info("Upload an OMR sheet image to see results here.")
            st.markdown("### Sample Data")
            if st.button("Load Sample Data"):
                load_sample_data()

def process_single_image(uploaded_file, answer_key_input, answer_key_format, subject_ranges, bubble_threshold, grade_a, grade_b, grade_c):
    """Process a single uploaded image"""
    try:
        # Convert uploaded file to OpenCV format
        image = Image.open(uploaded_file)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Parse answer key
        answer_key = parse_answer_key(answer_key_input, answer_key_format)
        if not answer_key:
            st.error("Invalid answer key format!")
            return
        
        # Parse subject ranges
        subjects = parse_subject_ranges(subject_ranges)
        if not subjects:
            st.error("Invalid subject ranges format!")
            return
        
        # Process the image
        with st.spinner("Processing OMR sheet..."):
            result = st.session_state.omr_processor.process_omr(
                image_cv, 
                answer_key, 
                subjects, 
                bubble_threshold
            )
        
        if result:
            display_results(result, grade_a, grade_b, grade_c)
        else:
            st.error("Failed to process the OMR sheet. Please check the image quality.")
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

def process_batch_folder(folder_path, answer_key_input, answer_key_format, subject_ranges, bubble_threshold, grade_a, grade_b, grade_c):
    """Process all images in a folder"""
    try:
        # Parse answer key and subjects
        answer_key = parse_answer_key(answer_key_input, answer_key_format)
        subjects = parse_subject_ranges(subject_ranges)
        
        if not answer_key or not subjects:
            st.error("Invalid configuration!")
            return
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(folder_path).glob(f"*{ext}"))
            image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
        
        if not image_files:
            st.warning("No image files found in the folder!")
            return
        
        st.write(f"Found {len(image_files)} images to process...")
        
        # Process each image
        results = []
        progress_bar = st.progress(0)
        
        for i, image_file in enumerate(image_files):
            try:
                # Load image
                image_cv = cv2.imread(str(image_file))
                if image_cv is None:
                    continue
                
                # Process
                result = st.session_state.omr_processor.process_omr(
                    image_cv, 
                    answer_key, 
                    subjects, 
                    bubble_threshold
                )
                
                if result:
                    result['filename'] = image_file.name
                    results.append(result)
                
                progress_bar.progress((i + 1) / len(image_files))
                
            except Exception as e:
                st.warning(f"Error processing {image_file.name}: {str(e)}")
        
        # Display batch results
        if results:
            display_batch_results(results, grade_a, grade_b, grade_c)
        else:
            st.error("No images were successfully processed!")
            
    except Exception as e:
        st.error(f"Error processing batch: {str(e)}")

def parse_answer_key(answer_key_input, format_type):
    """Parse answer key from different formats"""
    try:
        if format_type == "100-letter string":
            # Validate length and characters
            if len(answer_key_input) != 100:
                return None
            if not all(c in 'ABCD' for c in answer_key_input.upper()):
                return None
            return {str(i+1): answer_key_input[i].upper() for i in range(100)}
        else:
            # JSON format
            data = json.loads(answer_key_input)
            return {str(k): str(v).upper() for k, v in data.items()}
    except:
        return None

def parse_subject_ranges(subject_ranges):
    """Parse subject ranges string"""
    try:
        subjects = {}
        ranges = subject_ranges.split(',')
        for range_str in ranges:
            range_str = range_str.strip()
            if ':' in range_str:
                range_part, subject = range_str.split(':', 1)
                if '-' in range_part:
                    start, end = map(int, range_part.split('-'))
                    for i in range(start, end + 1):
                        subjects[str(i)] = subject.strip()
        return subjects
    except:
        return None

def display_results(result, grade_a, grade_b, grade_c):
    """Display processing results"""
    # Overall score
    total_score = result['total_score']
    percentage = result['percentage']
    grade = calculate_grade(percentage, grade_a, grade_b, grade_c)
    
    # Score display
    grade_class = f"grade-{grade}"
    st.markdown(f'<div class="score-display {grade_class}">Score: {total_score}/100 ({percentage:.1f}%) - Grade {grade}</div>', unsafe_allow_html=True)
    
    # Subject-wise scores
    st.subheader("📚 Subject-wise Scores")
    subject_data = []
    for subject, data in result['subjects'].items():
        subject_data.append({
            'Subject': subject,
            'Score': f"{data['score']}/{data['out_of']}",
            'Percentage': f"{(data['score']/data['out_of']*100):.1f}%"
        })
    
    df_subjects = pd.DataFrame(subject_data)
    st.dataframe(df_subjects, use_container_width=True)
    
    # Per-question results
    st.subheader("📋 Per-Question Results")
    question_data = []
    for q_num, data in result['per_question'].items():
        status = "✅" if data['correct'] else "❌" if data['valid'] else "⚠️"
        question_data.append({
            'Question': q_num,
            'Detected': data['detected'],
            'Correct': data['correct'],
            'Valid': data['valid'],
            'Status': status
        })
    
    df_questions = pd.DataFrame(question_data)
    st.dataframe(df_questions, use_container_width=True)
    
    # Annotated image
    if 'annotated_image' in result:
        st.subheader("🖼️ Annotated Image")
        st.image(result['annotated_image'], caption="Detected bubbles and markings", use_column_width=True)
    
    # Export options
    st.subheader("💾 Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        # JSON export
        json_data = json.dumps(result, indent=2)
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name="omr_results.json",
            mime="application/json"
        )
    
    with col2:
        # CSV export
        csv_data = df_questions.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="omr_results.csv",
            mime="text/csv"
        )

def display_batch_results(results, grade_a, grade_b, grade_c):
    """Display batch processing results"""
    st.subheader("📊 Batch Processing Results")
    
    # Summary table
    summary_data = []
    for result in results:
        grade = calculate_grade(result['percentage'], grade_a, grade_b, grade_c)
        summary_data.append({
            'Filename': result['filename'],
            'Score': f"{result['total_score']}/100",
            'Percentage': f"{result['percentage']:.1f}%",
            'Grade': grade
        })
    
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True)
    
    # Export batch results
    st.subheader("💾 Export Batch Results")
    batch_json = json.dumps(results, indent=2)
    st.download_button(
        label="Download Batch JSON",
        data=batch_json,
        file_name="batch_omr_results.json",
        mime="application/json"
    )

def calculate_grade(percentage, grade_a, grade_b, grade_c):
    """Calculate grade based on percentage"""
    if percentage >= grade_a:
        return "A"
    elif percentage >= grade_b:
        return "B"
    elif percentage >= grade_c:
        return "C"
    else:
        return "D"

def load_sample_data():
    """Load sample data for demonstration"""
    st.info("Sample data loading functionality would be implemented here.")
    st.write("This would load sample images and answer keys for testing.")

if __name__ == "__main__":
    main()
