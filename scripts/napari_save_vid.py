import napari
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter


lowestdicecell = '20240805_488_EGFP-CAAX_640_SPY650-DNA_cell3-04-Subset-07_frame_6.ome.tiff' #test
tiff_path = f'/opt/dlami/nvme/AvL/goodpred/test/{lowestdicecell}'

videoname = '20240805_488_EGFP-CAAX_640_SPY650-DNA_cell3-04-Subset-07_frame_6_lowestdice.gif'
gif_path = f'/mnt/efs/dlmbl/G-bs/AvL/predvid/{videoname}'

def save_gif_from_napari(image_data, gif_path, fps=15):
    # Create a Napari viewer and add the image data
    viewer = napari.Viewer()
    viewer.add_image(image_data, name='3D Image', colormap='gray')

    # Set up the GIF writer
    writer = PillowWriter(fps=fps)

    # Define the size of the GIF frame
    height, width = viewer.layers[0].data.shape[1:3]

    # Create a figure for matplotlib
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.axis('off')  # Hide axes

    # Start the GIF writer
    with writer.saving(fig, gif_path, dpi=100):
        # Iterate through each Z slice
        for z in range(image_data.shape[0]):
            viewer.dims.set_point(0, z)  # Move to the z-th slice
            
            # Get the current frame from the Napari viewer
            frame = viewer.screenshot()
            
            # Plot the frame with matplotlib
            ax.imshow(frame)
            
            # Write the frame to the GIF
            writer.grab_frame()

    print(f"GIF saved to {gif_path}")

# Load the 3D TIFF image
image_data = tiff.imread(tiff_path)[0,:,:,:]



# Save the animation as a GIF
save_gif_from_napari(image_data, gif_path, fps=15)