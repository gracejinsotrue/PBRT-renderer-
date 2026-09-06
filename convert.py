import OpenEXR
import Imath
import numpy as np
import cv2
import argparse
from pathlib import Path

def aces_tonemap(x):
    """Hill 2016 ACES approximation — matches Shaders.hlsl exactly."""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)

def convert_exr_to_png(exr_file, png_file, ev_compensation=0.0):
    print(f"Reading EXR with official OpenEXR library: {exr_file}")
    
    # 1. Verify file is valid
    if not OpenEXR.isOpenExrFile(str(exr_file)):
        raise ValueError(f"The file is not recognized as a valid EXR image.")
        
    # 2. Open EXR file and get header
    exr = OpenEXR.InputFile(str(exr_file))
    header = exr.header()
    
    # 3. Calculate resolution from the data window
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    # 4. Find the correct channel names 
    # (OpenCV often fails because it strictly expects R,G,B - but Nori sometimes uses r,g,b or single channels)
    channels = header['channels'].keys()
    
    if 'R' in channels and 'G' in channels and 'B' in channels:
        r_chan, g_chan, b_chan = 'R', 'G', 'B'
    elif 'r' in channels and 'g' in channels and 'b' in channels:
        r_chan, g_chan, b_chan = 'r', 'g', 'b'
    else:
        # Fallback just in case you render a grayscale depth map or AO map
        chan = list(channels)[0]
        print(f"Warning: Standard RGB channels not found. Using channel '{chan}' for all colors.")
        r_chan = g_chan = b_chan = chan

    # 5. Extract raw float data from channels
    # We ask for 32-bit floats, and OpenEXR automatically converts it if the file used 16-bit half-floats
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    r_str = exr.channel(r_chan, pt)
    g_str = exr.channel(g_chan, pt)
    b_str = exr.channel(b_chan, pt)

    # 6. Convert string data to numpy arrays
    r = np.frombuffer(r_str, dtype=np.float32).reshape(height, width)
    g = np.frombuffer(g_str, dtype=np.float32).reshape(height, width)
    b = np.frombuffer(b_str, dtype=np.float32).reshape(height, width)
    
    # 7. Stack into BGR format (OpenCV expects Blue-Green-Red for saving)
    img_linear = np.dstack([b, g, r])

    # 8. Replicate the shader pipeline: EV compensation → ACES → gamma 2.2
    img_linear = img_linear * pow(2.0, ev_compensation)
    img_tonemapped = aces_tonemap(img_linear)
    img_gamma = np.power(img_tonemapped, 1.0 / 2.2)
    img_8bit = (img_gamma * 255.0 + 0.5).astype(np.uint8)

    # 9. Save as PNG
    cv2.imwrite(str(png_file), img_8bit)
    print(f"Success! Saved to: {png_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert an EXR image to PNG.")
    parser.add_argument("input", help="The absolute or relative path to your .exr file")
    parser.add_argument("-o", "--output", help="Optional: The path to save the .png.", default=None)
    parser.add_argument("--ev", type=float, default=0.0,
                        help="EV compensation from scene XML (evCompensation). Default: 0.0")
    
    args = parser.parse_args()
    input_path = Path(args.input).resolve()
    
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = input_path.with_suffix('.png')
        
    convert_exr_to_png(input_path, output_path, ev_compensation=args.ev)