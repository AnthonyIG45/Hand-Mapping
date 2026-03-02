This project implements a semi-automated biometric system designed to extract unique feature vectors from hand images. 
By manually landmarking key points, the system calculates finger thickness and hand geometry to compare different samples using Euclidean distance.

Key Features
- Interactive Landmarking: A GUI interface allows users to click and define 12 specific points on the hand to guide feature extraction.
- Dynamic Thickness Calculation: * Automatically detects the "edges" of fingers by analyzing pixel intensity contrast.
- Calculates thickness along perpendicular axes at the midpoints of defined landmarks.
- Feature Vector Generation: Converts physical hand characteristics into a numerical array (vector) for each image.
- Biometric Comparison: Generates a distance matrix using Euclidean distance to compare the similarity between multiple hand samples.
- Visual Debugging: Renders "infinite" direction lines (blue) and measurement axes (red) to verify correct landmark placement.

Prerequisites
You will need the following Python packages:

       Bash
       pip install opencv-python numpy scipy

Usage
Prepare Images: Ensure you have 5 hand images named Hand_Image1.jpg through Hand_Image5.jpg in your directory.

Run the Script:

       Bash
       python hand_biometric.py

Landmarking Phase: * A window will open with the first image.
- Click 12 points in sequence (typically representing the base and tips of fingers or specific joints).
- Press any key once finished to process all images.

Results: The console will print the feature vectors for each hand and a $5 \times 5$ similarity matrix.
