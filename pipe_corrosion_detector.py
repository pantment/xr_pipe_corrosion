import cv2
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

class PipeCorrosionDetector:
    def __init__(self, image_path, output_path):
        # Load the original image and convert to grayscale
        self.ori_image = cv2.imread(image_path)
        self.output_path = output_path
        self.gray_image = cv2.cvtColor(self.ori_image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.gray_image.shape
        self.smoothed_image = None
        self.edges = None
        self.output_image = None
        self.last_black_pixels = None
        self.reference_line = None


    def preprocess_image(self):
        """Applies CLAHE and Gaussian smoothing to the image."""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(self.gray_image)
        
        # Apply Gaussian smoothing to reduce noise before edge detection
        self.smoothed_image = cv2.GaussianBlur(clahe_image, (5, 5), 0)
        cv2.imwrite(f"{self.output_path}clahe_smoothed_image.png", self.smoothed_image)

    def detect_edges(self, low_threshold=15, high_threshold=30):
        """Detects edges using the Canny edge detection."""
        # Perform Canny edge detection
        self.edges = cv2.Canny(self.smoothed_image, low_threshold, high_threshold)
        cv2.imwrite(f"{self.output_path}edges.png", self.edges)
        
        # Dilate and erode to adjust edge visibility
        kernel = np.ones((5, 5), np.uint8)
        self.edges = cv2.dilate(self.edges, kernel, iterations=5)
        self.edges = cv2.erode(self.edges, kernel, iterations=5)
        cv2.imwrite(f"{self.output_path}dilated_eroded_edges.png", self.edges)
        
        # Modify pixels below the first detected edge
        self.output_image = self.gray_image.copy()
        for col in range(self.width):
            for row in range(self.height):
                if self.edges[row, col] == 255:  # Edge detected at this point
                    self.output_image[row:, col] = 255
                    break
                else:
                    self.output_image[row, col] = 0
        
        # Perform a second round of Canny edge detection
        self.edges = cv2.Canny(self.output_image, 50, 150)
        cv2.imwrite(f"{self.output_path}output_image_edges.png", self.edges)

    def locate_last_edge(self):
        """Locate the last black pixel (pipe boundary) in each column."""
        self.last_black_pixels = []
        for x in range(self.width):
            column = self.edges[:, x]
            black_pixels = np.where(column == 255)[0]
            if len(black_pixels) > 0:
                self.last_black_pixels.append(black_pixels[-1])  # Get the lowest black pixel
            else:
                self.last_black_pixels.append(None)  # No black pixel found

    def draw_reference_line(self):
        """Calculate and draw the reference line based on the last black pixels."""
        valid_pixels = [y for y in self.last_black_pixels if y is not None]
        self.reference_line = int(np.max(valid_pixels))  # Maximum y-coordinate of the last black pixels

    def generate_heatmap(self, custom_cmap=None):
        """Generate and overlay a heatmap based on the distances from the reference line."""
        distances = []
        for y in self.last_black_pixels:
            if y is not None:
                distance = abs(y - self.reference_line)
                distances.append(distance)
            else:
                distances.append(0)  # No pipe found
        
        # Normalize the distances
        max_distance = max(distances)
        if max_distance > 0:
            normalized_distances = np.array(distances) / max_distance
        else:
            normalized_distances = np.zeros_like(distances)
        
        # Create a colormap transitioning from light blue -> blue -> yellow -> orange -> red
        if custom_cmap is None:
            colors = [
                (173/255, 216/255, 230/255),  # Light Blue (Low values)
                (0, 0, 1),    # Blue
                (1, 1, 0),    # Yellow
                (1, 165/255, 0),  # Orange
                (1, 0, 0)     # Red (High values)
            ]
        else:
            colors = custom_cmap
        custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
        heatmap = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Apply the heatmap color based on distances
        for x in range(self.width):
            distance_normalized = normalized_distances[x]
            color = custom_cmap(distance_normalized)  # Get colormap color
            color_rgb = (np.array(color[:3]) * 255).astype(np.uint8)  # Convert to RGB

            # Apply color to pixels between the last black pixel and reference line
            if self.last_black_pixels[x] is not None:
                y_start = min(self.last_black_pixels[x], self.reference_line)
                y_end = max(self.last_black_pixels[x], self.reference_line)
                heatmap[y_start:y_end, x, :] = color_rgb

        # Overlay heatmap onto the original image
        heatmap_mask = np.any(heatmap != [0, 0, 0], axis=-1)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        overlay = self.ori_image.copy()
        overlay[heatmap_mask] = cv2.addWeighted(overlay, 0.7, heatmap, 0.9, 1)[heatmap_mask]

        # Save output images
        cv2.imwrite(f"{self.output_path}heatmap.png", heatmap)
        cv2.imwrite(f"{self.output_path}overlay.png", overlay)

    def run(self):
        """Main function to run the entire pipeline."""
        self.preprocess_image()
        self.detect_edges()
        self.locate_last_edge()
        self.draw_reference_line()
        self.generate_heatmap()


# Example usage
img_path = ""
output_path = ""
detector = PipeCorrosionDetector(img_path, output_path)
detector.run()
