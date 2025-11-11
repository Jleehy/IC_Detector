import cv2
import numpy as np

# --- Setup to get the 'edges_closed' image ---
image = cv2.imread("b.png") 
output_image_final = image.copy() 
image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_filtered = cv2.bilateralFilter(image_grayscale, 9, 75, 75)
edges_canny = cv2.Canny(image_filtered, 30, 100)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
edges_closed = cv2.morphologyEx(edges_canny, cv2.MORPH_CLOSE, kernel)

height, width = edges_closed.shape

# --- 1. HORIZONTAL SCAN ---
scan_y = 150  ### <-- TUNE THIS VALUE ###
cv2.line(output_image_final, (0, scan_y), (width - 1, scan_y), (100, 100, 100), 2)

edge_transitions = []
in_white_region = False
for x in range(width):
    is_white = (edges_closed[scan_y, x] == 255)
    
    if is_white and not in_white_region:
        in_white_region = True
        edge_transitions.append(('left', x))
    elif not is_white and in_white_region:
        in_white_region = False
        edge_transitions.append(('right', x - 1))

# --- 2. FILTER OUT THE "JITTER" ---
min_gap_size =  50  ### <-- TUNE THIS VALUE ###
final_transitions = []
i = 0
while i < len(edge_transitions):
    if i == len(edge_transitions) - 1:
        final_transitions.append(edge_transitions[i])
        i += 1
        continue
    current_type, current_x = edge_transitions[i]
    next_type, next_x = edge_transitions[i + 1]
    
    if current_type == 'right' and next_type == 'left' and (next_x - current_x) < min_gap_size:
        i += 2
    else:
        final_transitions.append(edge_transitions[i])
        i += 1

# --- 3. STORE DOTS (TEMPORARY LIST) ---
dot_positions = [] 
for transition_type, x in final_transitions:
    dot_positions.append( (x, scan_y) )

# --- 4. AVERAGE CLOSE DOTS ---
averaging_threshold = 40 ### <-- TUNE THIS VALUE ###
averaged_dots = [] # This list has the (x,y) of the vertical edges

if len(dot_positions) > 0:
    current_group = [dot_positions[0]] 
    for i in range(1, len(dot_positions)):
        current_dot = dot_positions[i]
        last_dot_in_group = current_group[-1]
        
        if (current_dot[0] - last_dot_in_group[0]) < averaging_threshold:
            current_group.append(current_dot)
        else:
            sum_x = sum(dot[0] for dot in current_group)
            avg_x = int(round(sum_x / len(current_group)))
            averaged_dots.append( (avg_x, scan_y) )
            current_group = [current_dot]
    
    sum_x = sum(dot[0] for dot in current_group)
    avg_x = int(round(sum_x / len(current_group)))
    averaged_dots.append( (avg_x, scan_y) )

# --- Define Colors ---
green_color = (0, 255, 0) # Green color
red_color = (0, 0, 255)   # Red color
line_thickness = 2     # 2 pixels thick

# --- 5. VERTICAL "TIP" SCAN AND STORE LINE INFO ---
# We do this *before* drawing so we know which lines to make red.

print(f"\nFound {len(averaged_dots)} vertical edges.")
print("Starting vertical 'tip' scan...")
tip_dots = []
horizontal_lines = [] 
min_leftmost_x = width + 1 
leftmost_pin_x_coords = None # Will store (left_x, right_x) of the leftmost pin

# Loop through the vertical edge dots in pairs (left, right)
for i in range(0, len(averaged_dots), 2):
    if i + 1 < len(averaged_dots):
        left_edge_x = averaged_dots[i][0]
        right_edge_x = averaged_dots[i+1][0]
        
        lowest_tip_y = 0 
        lowest_tip_x = left_edge_x 
        
        for x in range(left_edge_x, right_edge_x + 1):
            for y in range(height - 1, 0, -1):
                if edges_closed[y, x] == 255:
                    if y > lowest_tip_y:
                        lowest_tip_y = y
                        lowest_tip_x = x
                    break 
        
        if lowest_tip_y > 0:
            tip_dots.append((lowest_tip_x, lowest_tip_y))
            horizontal_lines.append((left_edge_x, right_edge_x, lowest_tip_y))
            
            # Check if this pin is the new leftmost pin
            if left_edge_x < min_leftmost_x:
                min_leftmost_x = left_edge_x
                leftmost_pin_x_coords = (left_edge_x, right_edge_x) # Store both x values

print(f"Found {len(tip_dots)} pin tips.")
print(f"The absolute leftmost pin's x-coords are: {leftmost_pin_x_coords}")

# --- 6. DRAW ALL VERTICAL LINES ---
print("Drawing vertical lines...")
# Get the x-coords of the red lines, or (-1, -1) if none found
left_x, right_x = leftmost_pin_x_coords if leftmost_pin_x_coords else (-1, -1)

for (x, y) in averaged_dots:
    # If this x matches the left or right x of the leftmost pin, color it red
    if x == left_x or x == right_x:
        cv2.line(output_image_final, (x, 0), (x, height - 1), red_color, line_thickness)
    else:
        cv2.line(output_image_final, (x, 0), (x, height - 1), green_color, line_thickness)

# --- 7. DRAW ALL HORIZONTAL LINES ---
print("Drawing horizontal lines...")
for (x1, x2, y) in horizontal_lines:
    # Check if this line's x1 matches the leftmost x
    if x1 == min_leftmost_x:
        cv2.line(output_image_final, (x1, y), (x2, y), red_color, line_thickness)
    else:
        cv2.line(output_image_final, (x1, y), (x2, y), green_color, line_thickness)

# --- 8. DRAW THE TIP DOTS ---
dot_color = (0, 255, 0) # Green for dots
for (x, y) in tip_dots:
    cv2.circle(output_image_final, (x, y), 3, dot_color, -1)

# --- 9. DISPLAY FINAL RESULTS ---
cv2.imshow("Original B&W Edges (Map)", edges_closed)
cv2.imshow("Final Pin Edges and Tips", output_image_final)
cv2.waitKey(0)
cv2.destroyAllWindows()