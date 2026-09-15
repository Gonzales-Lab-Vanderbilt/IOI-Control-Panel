# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gonzales Lab, Vanderbilt University
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tkinter import Tk, filedialog

# Hide the small Tkinter root window
root = Tk()
root.withdraw()

path = filedialog.askopenfilename(
    title="Select image file",
    filetypes=[
        ("Image files", "*.tiff *.tif *.png *.jpg *.jpeg"),
        ("TIFF files", "*.tiff *.tif"),
        ("All files", "*.*"),
    ],
)

if not path:
    raise SystemExit("No file selected.")

img = np.array(Image.open(path))

print("Selected file:", path)
print("dtype:", img.dtype)
print("shape:", img.shape)
print("min:", img.min())
print("mean:", img.mean())
print("p50:", np.percentile(img, 50))
print("p99:", np.percentile(img, 99))
print("max:", img.max())

plt.imshow(img, cmap="gray")
plt.title("Raw linear display")
plt.axis("off")
plt.show()

plt.imshow(
    img,
    cmap="gray",
    vmin=np.percentile(img, 1),
    vmax=np.percentile(img, 99),
)
plt.title("Percentile-scaled display")
plt.axis("off")
plt.show()