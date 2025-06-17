Multi-Line Curve Tracer with Junction Awareness

This Python script processes images with thin curved lines (like wires, paths, or sketched strokes). It detects and traces each individual curve in the image, even when they intersect or overlap. The script uses vector-based logic to intelligently follow line directions through junctions and assigns a unique color to each detected path.

-----------------------------------------------------------------------------------------------------------------------------------------------

Features

Supports any number of lines within the image.
Uses direction-aware tracing to continue correctly through intersections.
Automatically detects and merges close junctions.
Assigns each line a different color.
Works for images with vertical, horizontal, or diagonal line orientations.
Input

A PNG or JPG image with dark lines on a light background.
The script converts the image to grayscale, binarizes it, and skeletonizes the result to extract the central path of each line.
Output

A colored visualization where:
Each distinct line is shown in a different color.
Untraced or disconnected pixels are shown in blue.
Junction areas are marked in purple.
Requirements

-----------------------------------------------------------------------------------------------------------------------------------------------

Install the required libraries using pip:

pip install numpy matplotlib opencv-python scikit-image

-----------------------------------------------------------------------------------------------------------------------------------------------

How to Run

Replace the image path inside the script:
image_path = "your_image.png"
Run the script:
python multiline_curve_tracer.py

-----------------------------------------------------------------------------------------------------------------------------------------------
Function Reference

get_neighbors(y, x, shape)
Returns all valid neighboring pixel coordinates (8-connectivity) for a given (y, x) position within the image bounds.

compute_segment_direction(segment)
Calculates the direction vector between the start and end of a list of coordinates. Used to determine the current tracing direction.

angle_between(v1, v2)
Computes the angle in degrees between two vectors. Used to evaluate which neighboring pixel continues the current direction most closely.

trace_direction_match(start, skeleton, junctions_set, window=15)
Performs directional tracing from a start pixel. It evaluates nearby candidates and chooses the next pixel based on angle alignment, avoiding revisiting and intelligently passing through junctions.

find_grouped_junctions(skeleton, radius=3)
Detects pixels that represent junctions (with more than 2 neighbors) and groups close junctions into single representative points based on a radius threshold.
