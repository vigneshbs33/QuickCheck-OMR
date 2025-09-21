# QuickCheck - OMR Evaluation System

**Developed by InteliCat Team**

QuickCheck is a robust, user-friendly OMR (Optical Mark Recognition) evaluation system that automatically processes scanned OMR sheets and provides detailed scoring and analysis.

## Features

- **Template-free Processing**: Automatically detects bubble positions without requiring predefined templates
- **Robust Orientation Correction**: Handles rotated, skewed, and upside-down OMR sheets
- **High Accuracy Detection**: Uses advanced image processing and clustering algorithms
- **Subject-wise Scoring**: Provides detailed breakdown by subject areas
- **Multiple Export Formats**: JSON and CSV result exports
- **Annotated Results**: Visual feedback showing detected selections and correctness
- **Batch Processing**: Process multiple images from a folder
- **Web Interface**: Easy-to-use Streamlit web application

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd QuickCheck
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Usage

### Web Interface

1. Open your browser and navigate to `http://localhost:8501`
2. Upload an OMR sheet image (JPEG/PNG)
3. Enter the answer key in one of two formats:
   - **100-letter string**: e.g., "ABCDABCD..."
   - **JSON format**: `{"1": "A", "2": "B", ...}`
4. Specify subject ranges: e.g., "1-20:Python,21-40:EDA,41-60:MySQL,61-80:PowerBI,81-100:AdvStats"
5. Click "Evaluate" to process the OMR sheet

### Batch Processing

1. Place your OMR images in a folder
2. Enter the folder path in the "Batch Processing" section
3. Click "Process Batch Folder" to evaluate all images

### Command Line Testing

```bash
python test_sample_data.py
```

## OMR Format Support

QuickCheck is optimized for OMR sheets with:
- 100 questions total
- 5 subject columns (20 questions each)
- 4 options per question (A, B, C, D)
- Standard bubble format

## Configuration

### Answer Key Formats

**100-letter string**:
```
ABCDABCDABCD...
```

**JSON format**:
```json
{
  "1": "A",
  "2": "B",
  "3": "C",
  "4": "D",
  ...
}
```

### Subject Ranges

Format: `start-end:SubjectName,start-end:SubjectName,...`

Example:
```
1-20:Python,21-40:Data Analysis,41-60:MySQL,61-80:Power BI,81-100:Adv Stats
```

## Output Format

The system generates comprehensive results including:

- **Total Score**: Overall score out of 100
- **Percentage**: Percentage score
- **Grade**: Letter grade (A, B, C, D)
- **Subject-wise Scores**: Breakdown by subject
- **Per-question Results**: Detailed analysis of each question
- **Annotated Image**: Visual representation of detected selections

### Sample Output

```json
{
  "student_id": null,
  "total_score": 76,
  "percentage": 76.0,
  "grade": "B",
  "subjects": {
    "Python": {"score": 18, "out_of": 20},
    "Data Analysis": {"score": 14, "out_of": 20},
    "MySQL": {"score": 16, "out_of": 20},
    "Power BI": {"score": 15, "out_of": 20},
    "Adv Stats": {"score": 13, "out_of": 20}
  },
  "per_question": {
    "1": {"detected": "A", "valid": true, "correct": true},
    "2": {"detected": "B", "valid": true, "correct": false},
    ...
  }
}
```

## Technical Details

### Image Processing Pipeline

1. **Orientation Correction**: Automatic detection and correction of rotated sheets
2. **Bubble Detection**: Advanced contour detection and clustering
3. **Position Mapping**: Template-based position mapping with adaptive scaling
4. **Filled Detection**: Multi-metric analysis for accurate bubble filling detection
5. **Evaluation**: Comprehensive scoring and validation

### Dependencies

- **streamlit**: Web interface
- **opencv-python**: Image processing
- **numpy**: Numerical operations
- **pillow**: Image manipulation
- **pandas**: Data processing
- **scikit-learn**: Machine learning algorithms

## File Structure

```
QuickCheck/
├── app.py                          # Main Streamlit application
├── omr/
│   ├── __init__.py
│   ├── adaptive_engine.py          # Main OMR processing engine
│   ├── accurate_engine.py          # Template-based engine
│   ├── calibrated_engine.py        # Calibration-based engine
│   └── template_engine.py          # Fixed template engine
├── sample_data/
│   ├── sample_answer_key.json      # Sample answer key
│   ├── test_image.jpeg            # Sample test images
│   └── test_results.json          # Test results
├── requirements.txt                # Python dependencies
├── test_sample_data.py            # Testing script
└── README.md                      # This file
```

## Troubleshooting

### Low Detection Accuracy

1. Ensure good image quality (high resolution, good lighting)
2. Check that the OMR sheet is not heavily skewed
3. Verify the answer key format is correct
4. Try adjusting the bubble threshold in the UI

### Processing Errors

1. Check that the image format is supported (JPEG, PNG)
2. Ensure the image contains a valid OMR sheet
3. Verify the subject ranges format is correct

## License

This implementation is provided to InteliCat for exclusive use under the project name QuickCheck.

## Support

For technical support or questions, please contact the InteliCat development team.

---

**QuickCheck - Making OMR evaluation simple and accurate!**
