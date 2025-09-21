"""
QuickCheck - Advanced OMR Evaluation System
Developed by InteliCat Team

A comprehensive, template-free OMR evaluation system optimized for accuracy and ease of use.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
from PIL import Image
import io
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import our refined OMR engine
from omr.refined_engine import RefinedOMRProcessor

# Page configuration
st.set_page_config(
    page_title="QuickCheck - OMR Evaluator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
    <style>
    .main-header {
        font-size: 4em;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 0.2em;
        text-shadow: 2px 2px 4px #aaaaaa;
        background: linear-gradient(45deg, #2E8B57, #32CD32);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5em;
        color: #555555;
        text-align: center;
        margin-bottom: 1em;
    }
    .intelicat-brand {
        color: #FF6B35;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-card {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stButton>button {
        background: linear-gradient(45deg, #2E8B57, #32CD32);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 1.1em;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .stSelectbox>div>div>select {
        border-radius: 8px;
    }
    .stTextArea>div>div>textarea {
        border-radius: 8px;
    }
    .stFileUploader>div>div>div>div {
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

def load_answer_key_from_json(uploaded_file) -> Optional[Dict[str, str]]:
    """Load answer key from uploaded JSON file"""
    try:
        content = uploaded_file.read()
        data = json.loads(content.decode('utf-8'))
        
        # Handle different JSON formats
        if 'subjects' in data:
            # Format: {"subjects": {"python": {"questions": {"1": "A", ...}}}}
            answer_key = {}
            for subject_data in data['subjects'].values():
                if 'questions' in subject_data:
                    answer_key.update(subject_data['questions'])
            return answer_key
        else:
            # Format: {"1": "A", "2": "B", ...}
            return data
    except Exception as e:
        st.error(f"Error loading JSON file: {str(e)}")
        return None

def parse_subject_ranges(subject_ranges_str: str) -> Dict[str, Dict[str, int]]:
    """Parse subject ranges string into dictionary"""
    subjects = {}
    try:
        for entry in subject_ranges_str.split(','):
            if ':' in entry:
                range_str, subject_name = entry.split(':', 1)
                range_str = range_str.strip()
                subject_name = subject_name.strip()
                
                if '-' in range_str:
                    start, end = map(int, range_str.split('-'))
                    subjects[subject_name] = {'start': start, 'end': end}
                else:
                    st.warning(f"Invalid range format: {range_str}. Skipping.")
    except Exception as e:
        st.error(f"Error parsing subject ranges: {str(e)}")
    return subjects

def create_sample_answer_key() -> str:
    """Create a sample answer key for demonstration"""
    sample_key = {}
    for i in range(1, 101):
        # Create a pattern: A, B, C, D, A, B, C, D...
        options = ['A', 'B', 'C', 'D']
        sample_key[str(i)] = options[(i - 1) % 4]
    return json.dumps(sample_key, indent=2)

def main():
    """Main application function"""
    # Header
    st.markdown('<h1 class="main-header">QuickCheck</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced OMR Evaluation System by <span class="intelicat-brand">InteliCat</span></p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'omr_processor' not in st.session_state:
        st.session_state.omr_processor = RefinedOMRProcessor()
    
    if 'processed_results' not in st.session_state:
        st.session_state.processed_results = None
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Answer Key Input
        st.subheader("📝 Answer Key")
        answer_key_type = st.radio(
            "Select Answer Key Input Type:", 
            ("JSON Upload", "Manual Entry"),
            help="Upload a JSON file or manually enter the answer key"
        )
        
        answer_key_input = None
        if answer_key_type == "JSON Upload":
            answer_key_file = st.file_uploader(
                "Upload Answer Key JSON:", 
                type=["json"],
                help="Upload a JSON file with answer key in format: {\"1\": \"A\", \"2\": \"B\", ...}"
            )
            if answer_key_file is not None:
                answer_key_input = load_answer_key_from_json(answer_key_file)
        else:
            # Manual entry
            st.text_area(
                "Enter Answer Key (JSON format):",
                value=create_sample_answer_key(),
                height=200,
                help="Enter answer key in JSON format: {\"1\": \"A\", \"2\": \"B\", ...}"
            )
            try:
                answer_key_input = json.loads(st.session_state.get('answer_key_text', create_sample_answer_key()))
            except:
                st.warning("Please enter valid JSON format")
        
        # Subject Ranges Input
        st.subheader("📚 Subject Ranges")
        subject_ranges_input = st.text_area(
            "Enter Subject Ranges:",
            value="1-20:Python,21-40:EDA,41-60:SQL,61-80:Power BI,81-100:Statistics",
            height=100,
            help="Format: 1-20:Subject1,21-40:Subject2,..."
        )
        
        # Processing Parameters
        st.subheader("🔧 Processing Parameters")
        bubble_threshold = st.slider(
            "Bubble Detection Threshold:",
            min_value=0.1,
            max_value=0.9,
            value=0.4,
            step=0.05,
            help="Lower values detect darker bubbles (more sensitive)"
        )
        
        # Grade Thresholds
        st.subheader("📊 Grade Thresholds")
        col1, col2 = st.columns(2)
        with col1:
            grade_a = st.number_input("Grade A (>= %):", min_value=0, max_value=100, value=85)
            grade_b = st.number_input("Grade B (>= %):", min_value=0, max_value=100, value=70)
        with col2:
            grade_c = st.number_input("Grade C (>= %):", min_value=0, max_value=100, value=50)
        
        grade_thresholds = {
            'A': grade_a,
            'B': grade_b,
            'C': grade_c,
            'D': 0
        }
        
        # System Info
        st.subheader("ℹ️ System Info")
        st.info("""
        **QuickCheck Features:**
        - Template-free processing
        - Automatic orientation correction
        - Advanced bubble detection
        - Subject-wise analysis
        - Batch processing support
        """)

    # Main content area
    st.header("📸 Upload OMR Sheet")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an OMR sheet image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of the OMR sheet"
    )
    
    if uploaded_file is not None:
        # Read and display image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image, caption="Uploaded OMR Sheet", use_column_width=True)
        with col2:
            st.info(f"""
            **Image Details:**
            - Size: {image.shape[1]} x {image.shape[0]} pixels
            - Format: {uploaded_file.type}
            - File: {uploaded_file.name}
            """)
        
        # Process button
        if st.button("🎯 Evaluate OMR Sheet", type="primary"):
            if not answer_key_input or not subject_ranges_input:
                st.error("❌ Please provide both Answer Key and Subject Ranges.")
            else:
                with st.spinner("🔄 Processing OMR sheet..."):
                    try:
                        # Parse subject ranges
                        subjects = parse_subject_ranges(subject_ranges_input)
                        
                        # Process OMR sheet
                        results, annotated_image = st.session_state.omr_processor.process_omr_sheet(
                            image, answer_key_input, subjects, bubble_threshold, grade_thresholds
                        )
                        
                        if results and annotated_image is not None:
                            st.session_state.processed_results = (results, annotated_image)
                            st.success("✅ OMR Sheet Evaluated Successfully!")
                        else:
                            st.error("❌ Failed to evaluate OMR sheet. Please check the image and configuration.")
                            
                    except Exception as e:
                        st.error(f"❌ An error occurred during evaluation: {str(e)}")
                        st.exception(e)
    
    # Display results if available
    if st.session_state.processed_results:
        results, annotated_image = st.session_state.processed_results
        
        st.header("📊 Evaluation Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Score", f"{results['total_score']}/100")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Percentage", f"{results['percentage']:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Grade", results['grade'])
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            detected_count = len([k for k, v in results['per_question'].items() if v['detected'] != 'None'])
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Questions Detected", f"{detected_count}/100")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Subject-wise scores
        st.subheader("📚 Subject-wise Scores")
        subject_data = []
        for subject, data in results['subjects'].items():
            percentage = (data['score'] / data['out_of']) * 100 if data['out_of'] > 0 else 0
            subject_data.append({
                "Subject": subject,
                "Score": data['score'],
                "Out of": data['out_of'],
                "Percentage": f"{percentage:.1f}%"
            })
        
        subject_df = pd.DataFrame(subject_data)
        st.dataframe(subject_df, use_container_width=True)
        
        # Per-question details
        st.subheader("📝 Per Question Details")
        question_data = []
        for q_num, q_data in results['per_question'].items():
            question_data.append({
                "Question": q_num,
                "Detected": q_data.get("detected", "N/A"),
                "Correct": "✅" if q_data.get("correct", False) else "❌",
                "Valid": "✅" if q_data.get("valid", True) else "❌"
            })
        
        question_df = pd.DataFrame(question_data)
        st.dataframe(question_df, use_container_width=True)
        
        # Annotated image
        st.subheader("🖼️ Annotated OMR Sheet")
        st.image(annotated_image, caption="Annotated OMR Sheet with Detected Selections", use_column_width=True)
        
        # Download options
        st.subheader("💾 Download Results")
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON download
            results_json = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="📄 Download JSON Results",
                data=results_json,
                file_name="quickcheck_results.json",
                mime="application/json",
                help="Download complete results in JSON format"
            )
        
        with col2:
            # CSV download
            results_csv = pd.DataFrame([results]).to_csv(index=False)
            st.download_button(
                label="📊 Download CSV Results",
                data=results_csv,
                file_name="quickcheck_results.csv",
                mime="text/csv",
                help="Download results in CSV format for spreadsheet analysis"
            )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p><strong>QuickCheck</strong> - Advanced OMR Evaluation System</p>
            <p>Developed by <span style='color: #FF6B35; font-weight: bold;'>InteliCat Team</span></p>
            <p>Making OMR evaluation simple, accurate, and efficient</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()