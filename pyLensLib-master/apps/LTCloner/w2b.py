from PIL import Image, ImageOps
import numpy as np
import argparse

def selective_white_to_black(input_path, output_path, white_threshold=240, black_threshold=50):
    """
    Convert white background to black while preserving axes, labels, and colored content.
    This version is optimized for scientific plots and figures.
    
    Args:
        input_path: Path to input image
        output_path: Path to save output image
        white_threshold: RGB values above this are considered "white" (0-255)
        black_threshold: RGB values below this are considered "black" and preserved (0-255)
    """
    img = Image.open(input_path).convert("RGB")
    img_array = np.array(img)
    
    # Create a copy for the output
    output_array = img_array.copy()
    
    # Calculate brightness and color properties
    brightness = np.mean(img_array, axis=2)
    color_range = np.max(img_array, axis=2) - np.min(img_array, axis=2)
    
    # Find white and very light pixels to convert to black
    white_mask = np.all(img_array >= white_threshold, axis=2)
    
    # Find very light gray pixels (high brightness, low color variation)
    light_gray_mask = (brightness >= white_threshold * 0.95) & (color_range <= 15)
    
    # Combine masks for pixels to convert to black
    to_black_mask = white_mask | light_gray_mask
    
    # Convert selected pixels to black
    output_array[to_black_mask] = [0, 0, 0]
    
    # Keep everything else unchanged (including gray axes, black text, colors)
    
    # Convert back to PIL Image and save
    result_img = Image.fromarray(output_array)
    result_img.save(output_path)
    
    converted_pixels = np.sum(to_black_mask)
    total_pixels = img_array.shape[0] * img_array.shape[1]
    unchanged_pixels = total_pixels - converted_pixels
    
    print(f"Processed image: {img_array.shape[1]}x{img_array.shape[0]} pixels")
    print(f"  - White/light pixels converted to black: {converted_pixels}")
    print(f"  - Pixels preserved (text, axes, colors): {unchanged_pixels}")
    print(f"  - Percentage preserved: {unchanged_pixels/total_pixels*100:.1f}%")
    print(f"Saved processed image to {output_path}")

def simple_white_to_black(input_path, output_path, white_threshold=240):
    """
    Simple version: only convert white pixels to black, keep everything else unchanged.
    
    Args:
        input_path: Path to input image
        output_path: Path to save output image
        white_threshold: RGB values above this are considered "white" (0-255)
    """
    img = Image.open(input_path).convert("RGB")
    img_array = np.array(img)
    
    # Create a copy for the output
    output_array = img_array.copy()
    
    # Find white pixels (all RGB channels above threshold)
    white_mask = np.all(img_array >= white_threshold, axis=2)
    
    # Convert white pixels to black
    output_array[white_mask] = [0, 0, 0]
    
    # Convert back to PIL Image and save
    result_img = Image.fromarray(output_array)
    result_img.save(output_path)
    
    white_pixels = np.sum(white_mask)
    total_pixels = img_array.shape[0] * img_array.shape[1]
    
    print(f"Processed image: {img_array.shape[1]}x{img_array.shape[0]} pixels")
    print(f"  - White pixels converted to black: {white_pixels}")
    print(f"  - Other pixels unchanged: {total_pixels - white_pixels}")
    print(f"Saved processed image to {output_path}")

def plot_friendly_conversion(input_path, output_path, background_threshold=250):
    """
    Plot-friendly conversion: designed specifically for scientific plots and figures.
    Handles anti-aliased text and removes white contours around plot elements.
    
    Args:
        input_path: Path to input image
        output_path: Path to save output image
        background_threshold: Only pixels brighter than this become black (default: 250)
    """
    img = Image.open(input_path).convert("RGB")
    img_array = np.array(img)
    
    # Create a copy for the output
    output_array = img_array.copy()
    
    # Calculate brightness for each pixel
    brightness = np.mean(img_array, axis=2)
    color_range = np.max(img_array, axis=2) - np.min(img_array, axis=2)
    
    # Find very bright pixels (pure background)
    pure_white_mask = np.all(img_array >= background_threshold, axis=2)
    
    # Find light pixels with low color variation (light gray anti-aliasing)
    light_antialiasing_mask = (brightness >= background_threshold - 15) & (color_range <= 10)
    
    # Find pixels that are very bright overall (catches light contours)
    bright_contour_mask = brightness >= background_threshold - 8
    
    # Combine masks: convert background + light anti-aliasing + bright contours
    to_black_mask = pure_white_mask | (light_antialiasing_mask & bright_contour_mask)
    
    # Convert selected pixels to black
    output_array[to_black_mask] = [0, 0, 0]
    
    # Convert back to PIL Image and save
    result_img = Image.fromarray(output_array)
    result_img.save(output_path)
    
    converted_pixels = np.sum(to_black_mask)
    total_pixels = img_array.shape[0] * img_array.shape[1]
    unchanged_pixels = total_pixels - converted_pixels
    
    print(f"Plot-friendly conversion completed:")
    print(f"  - Image size: {img_array.shape[1]}x{img_array.shape[0]} pixels")
    print(f"  - Background + contours converted to black: {converted_pixels}")
    print(f"  - Content preserved (axes, labels, data): {unchanged_pixels}")
    print(f"  - Preservation rate: {unchanged_pixels/total_pixels*100:.1f}%")
    print(f"Saved to {output_path}")

def smart_plot_conversion(input_path, output_path, background_threshold=252):
    """
    Intelligent conversion that preserves plot structure (axes, labels, data).
    Uses edge detection and morphological analysis to identify plot elements.
    
    Args:
        input_path: Path to input image
        output_path: Path to save output image  
        background_threshold: Threshold for background pixels (default: 252)
    """
    from scipy import ndimage
    
    img = Image.open(input_path).convert("RGB")
    img_array = np.array(img)
    
    # Create a copy for the output
    output_array = img_array.copy()
    
    # Calculate brightness
    brightness = np.mean(img_array, axis=2)
    
    # Find potential plot elements using edge detection
    # Convert to grayscale for edge detection
    gray = np.mean(img_array, axis=2).astype(np.uint8)
    
    # Detect edges (likely axes, text, data points)
    from scipy.ndimage import sobel
    edges_x = sobel(gray, axis=0)
    edges_y = sobel(gray, axis=1)
    edges = np.sqrt(edges_x**2 + edges_y**2)
    
    # Create protection mask for areas near edges (plot elements)
    edge_threshold = np.percentile(edges, 90)  # Top 10% of edge strength
    strong_edges = edges > edge_threshold
    
    # Dilate edge areas to protect nearby pixels
    protection_mask = ndimage.binary_dilation(strong_edges, iterations=2)
    
    # Identify background: bright pixels that are NOT near plot elements
    background_candidates = brightness >= background_threshold
    background_mask = background_candidates & ~protection_mask
    
    # Convert background pixels to black
    output_array[background_mask] = [0, 0, 0]
    
    # Convert back to PIL Image and save
    result_img = Image.fromarray(output_array)
    result_img.save(output_path)
    
    converted_pixels = np.sum(background_mask)
    protected_pixels = np.sum(protection_mask)
    total_pixels = img_array.shape[0] * img_array.shape[1]
    unchanged_pixels = total_pixels - converted_pixels
    
    print(f"Smart plot conversion completed:")
    print(f"  - Image size: {img_array.shape[1]}x{img_array.shape[0]} pixels")
    print(f"  - Background threshold: {background_threshold}")
    print(f"  - Plot elements protected: {protected_pixels}")
    print(f"  - Background converted to black: {converted_pixels}")
    print(f"  - Total preserved: {unchanged_pixels}")
    print(f"  - Preservation rate: {unchanged_pixels/total_pixels*100:.1f}%")
    print(f"Saved to {output_path}")

def gradient_aware_conversion(input_path, output_path, bg_threshold=250, gradient_protection=True):
    """
    Conversion that uses gradient information to avoid destroying plot elements.
    Safer alternative that doesn't require scipy.
    
    Args:
        input_path: Path to input image
        output_path: Path to save output image  
        bg_threshold: Threshold for background pixels (default: 250)
        gradient_protection: Whether to protect areas with high gradients
    """
    img = Image.open(input_path).convert("RGB")
    img_array = np.array(img)
    
    # Create a copy for the output
    output_array = img_array.copy()
    
    # Calculate brightness
    brightness = np.mean(img_array, axis=2)
    
    if gradient_protection:
        # Calculate simple gradients to identify plot elements
        grad_x = np.abs(np.diff(brightness, axis=1, prepend=brightness[:, :1]))
        grad_y = np.abs(np.diff(brightness, axis=0, prepend=brightness[:1, :]))
        gradient_magnitude = grad_x + grad_y
        
        # Protect areas with high gradients (edges, text, axes)
        gradient_threshold = np.percentile(gradient_magnitude, 85)
        high_gradient_mask = gradient_magnitude > gradient_threshold
        
        # Only convert bright pixels that are NOT in high-gradient areas
        background_mask = (brightness >= bg_threshold) & ~high_gradient_mask
    else:
        # Simple brightness-only approach
        background_mask = brightness >= bg_threshold
    
    # Convert background pixels to black
    output_array[background_mask] = [0, 0, 0]
    
    # Convert back to PIL Image and save
    result_img = Image.fromarray(output_array)
    result_img.save(output_path)
    
    converted_pixels = np.sum(background_mask)
    total_pixels = img_array.shape[0] * img_array.shape[1]
    unchanged_pixels = total_pixels - converted_pixels
    
    print(f"Gradient-aware conversion completed:")
    print(f"  - Image size: {img_array.shape[1]}x{img_array.shape[0]} pixels")
    print(f"  - Background threshold: {bg_threshold}")
    print(f"  - Gradient protection: {gradient_protection}")
    print(f"  - Background converted to black: {converted_pixels}")
    print(f"  - Content preserved: {unchanged_pixels}")
    print(f"  - Preservation rate: {unchanged_pixels/total_pixels*100:.1f}%")
    print(f"Saved to {output_path}")

def anti_alias_aware_conversion(input_path, output_path, bg_threshold=252, aa_threshold=230):
    """
    Specifically designed to handle anti-aliased plots with white contours.
    Uses a multi-pass approach to remove contours while preserving plot structure.
    
    Args:
        input_path: Path to input image
        output_path: Path to save output image  
        bg_threshold: Main background threshold (default: 252)
        aa_threshold: Anti-aliasing contour threshold (default: 230)
    """
    img = Image.open(input_path).convert("RGB")
    img_array = np.array(img)
    
    # Create a copy for the output
    output_array = img_array.copy()
    
    # Calculate brightness and color properties
    brightness = np.mean(img_array, axis=2)
    color_range = np.max(img_array, axis=2) - np.min(img_array, axis=2)
    
    # Step 1: Find the core content (very dark pixels that should definitely stay)
    core_content = brightness <= 100  # Very dark pixels (text, data, axes cores)
    
    # Step 2: Find pure background (very bright pixels)
    pure_background = brightness >= bg_threshold
    
    # Step 3: Find anti-aliasing candidates (medium-bright, low color variation)
    aa_candidates = (brightness >= aa_threshold) & (brightness < bg_threshold) & (color_range <= 20)
    
    # Step 4: More sophisticated approach - look at neighborhood context
    # Create a simple "distance from dark content" measure
    h, w = brightness.shape
    
    # Find pixels that are "isolated" bright pixels (likely contours)
    # by checking if they have dark neighbors
    isolated_bright = np.zeros_like(brightness, dtype=bool)
    
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            
            # Shift the core content mask to check neighbors
            y_start, y_end = max(0, -dy), min(h, h - dy)
            x_start, x_end = max(0, -dx), min(w, w - dx)
            
            shifted_y_start, shifted_y_end = max(0, dy), min(h, h + dy)
            shifted_x_start, shifted_x_end = max(0, dx), min(w, w + dx)
            
            # Check if bright pixels have dark neighbors
            neighbor_check = np.zeros_like(brightness, dtype=bool)
            neighbor_check[y_start:y_end, x_start:x_end] = core_content[shifted_y_start:shifted_y_end, shifted_x_start:shifted_x_end]
            
            # Mark bright pixels that have dark neighbors as "isolated bright" (likely contours)
            isolated_bright |= aa_candidates & neighbor_check
    
    # Combine all pixels to convert to black
    to_black_mask = pure_background | isolated_bright
    
    # Convert selected pixels to black
    output_array[to_black_mask] = [0, 0, 0]
    
    # Convert back to PIL Image and save
    result_img = Image.fromarray(output_array)
    result_img.save(output_path)
    
    background_pixels = np.sum(pure_background)
    contour_pixels = np.sum(isolated_bright)
    core_pixels = np.sum(core_content)
    total_converted = np.sum(to_black_mask)
    total_pixels = img_array.shape[0] * img_array.shape[1]
    unchanged_pixels = total_pixels - total_converted
    
    print(f"Anti-alias aware conversion completed:")
    print(f"  - Image size: {img_array.shape[1]}x{img_array.shape[0]} pixels")
    print(f"  - Background threshold: {bg_threshold}, AA threshold: {aa_threshold}")
    print(f"  - Core content pixels (protected): {core_pixels}")
    print(f"  - Pure background converted: {background_pixels}")
    print(f"  - Anti-aliasing contours removed: {contour_pixels}")
    print(f"  - Total converted to black: {total_converted}")
    print(f"  - Content preserved: {unchanged_pixels}")
    print(f"  - Preservation rate: {unchanged_pixels/total_pixels*100:.1f}%")
    print(f"Saved to {output_path}")

def morphological_conversion(input_path, output_path, bg_threshold=250, erosion_size=1):
    """
    Uses morphological operations to clean up white contours.
    First converts background, then erodes white regions to remove thin contours.
    
    Args:
        input_path: Path to input image
        output_path: Path to save output image  
        bg_threshold: Background threshold (default: 250)
        erosion_size: Size of erosion kernel to remove thin contours (default: 1)
    """
    img = Image.open(input_path).convert("RGB")
    img_array = np.array(img)
    
    # Create a copy for the output
    output_array = img_array.copy()
    
    # Calculate brightness
    brightness = np.mean(img_array, axis=2)
    
    # Step 1: Convert obvious background
    background_mask = brightness >= bg_threshold
    output_array[background_mask] = [0, 0, 0]
    
    # Step 2: Find remaining white/light pixels (potential contours)
    remaining_brightness = np.mean(output_array, axis=2)
    light_pixels = remaining_brightness >= 200  # Still quite bright
    
    # Step 3: Apply morphological erosion to remove thin white contours
    # Simple erosion without scipy
    eroded_light = light_pixels.copy()
    
    for _ in range(erosion_size):
        new_eroded = eroded_light.copy()
        h, w = eroded_light.shape
        
        for y in range(1, h-1):
            for x in range(1, w-1):
                # If any neighbor is dark, erode this pixel
                if eroded_light[y, x] and not np.all(eroded_light[y-1:y+2, x-1:x+2]):
                    new_eroded[y, x] = False
        
        eroded_light = new_eroded
    
    # Convert eroded light pixels to black (these are likely contours)
    contour_pixels = light_pixels & ~eroded_light
    output_array[contour_pixels] = [0, 0, 0]
    
    # Convert back to PIL Image and save
    result_img = Image.fromarray(output_array)
    result_img.save(output_path)
    
    background_pixels = np.sum(background_mask)
    contour_removed = np.sum(contour_pixels)
    total_converted = background_pixels + contour_removed
    total_pixels = img_array.shape[0] * img_array.shape[1]
    unchanged_pixels = total_pixels - total_converted
    
    print(f"Morphological conversion completed:")
    print(f"  - Image size: {img_array.shape[1]}x{img_array.shape[0]} pixels")
    print(f"  - Background threshold: {bg_threshold}")
    print(f"  - Background pixels converted: {background_pixels}")
    print(f"  - Contour pixels removed: {contour_removed}")
    print(f"  - Total converted to black: {total_converted}")
    print(f"  - Content preserved: {unchanged_pixels}")
    print(f"  - Preservation rate: {unchanged_pixels/total_pixels*100:.1f}%")
    print(f"Saved to {output_path}")

def invert_white_to_black(input_path, output_path):
    """
    Legacy function: Invert a white-background image to black, flipping all colors.
    Works by inverting all RGB channels.
    """
    img = Image.open(input_path).convert("RGB")

    # Invert the RGB channels
    inverted = ImageOps.invert(img)

    # Optional: if the background becomes grayish instead of deep black,
    # you can enhance contrast or apply thresholding:
    inverted = ImageOps.autocontrast(inverted, cutoff=0)

    inverted.save(output_path)
    print(f"Saved inverted image to {output_path}")

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert white background to black while preserving other colors."
    )
    parser.add_argument("input", help="Path to the input image file.")
    parser.add_argument("output", help="Path to save the output image file.")
    parser.add_argument(
        "--white-threshold", 
        type=int, 
        default=240, 
        help="RGB threshold for white pixels (0-255, default: 240)"
    )
    parser.add_argument(
        "--black-threshold", 
        type=int, 
        default=50, 
        help="RGB threshold for black pixels (0-255, default: 50)"
    )
    parser.add_argument(
        "--simple", 
        action="store_true",
        help="Simple mode: only convert white pixels to black"
    )
    parser.add_argument(
        "--plot", 
        action="store_true",
        help="Plot-friendly mode: optimized for scientific plots and figures"
    )
    parser.add_argument(
        "--smart", 
        action="store_true",
        help="Smart mode: uses edge detection to preserve plot elements (requires scipy)"
    )
    parser.add_argument(
        "--gradient", 
        action="store_true",
        help="Gradient-aware mode: protects high-gradient areas (axes, text)"
    )
    parser.add_argument(
        "--anti-alias", 
        action="store_true",
        help="Anti-alias mode: removes white contours using neighbor analysis"
    )
    parser.add_argument(
        "--morphological", 
        action="store_true",
        help="Morphological mode: uses erosion to remove thin contours"
    )
    parser.add_argument(
        "--contour-threshold",
        type=int,
        default=230,
        help="Threshold for removing anti-aliasing contours (0-255, default: 230)"
    )
    parser.add_argument(
        "--legacy", 
        action="store_true",
        help="Use legacy inversion method (inverts all colors)"
    )
    
    args = parser.parse_args()
    
    if args.legacy:
        invert_white_to_black(args.input, args.output)
    elif args.simple:
        simple_white_to_black(args.input, args.output, args.white_threshold)
    elif args.plot:
        plot_friendly_conversion(args.input, args.output, args.white_threshold)
    elif args.smart:
        smart_plot_conversion(args.input, args.output, args.white_threshold)
    elif args.gradient:
        gradient_aware_conversion(args.input, args.output, args.white_threshold, True)
    elif args.anti_alias:
        anti_alias_aware_conversion(args.input, args.output, args.white_threshold, args.contour_threshold)
    elif args.morphological:
        morphological_conversion(args.input, args.output, args.white_threshold, 1)
    else:
        selective_white_to_black(
            args.input, 
            args.output, 
            args.white_threshold, 
            args.black_threshold
        )