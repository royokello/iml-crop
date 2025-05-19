"""
Size conversion module for IML Crop.
Maps between crop shape text descriptions and integer class values.
"""

# Dictionary mapping from crop shape text to integer class
SHAPE_TO_INT = {
    # Squares (1:1)
    "64x64 Square": 1,
    "128x128 Square": 2,
    "192x192 Square": 3,
    "256x256 Square": 4,
    "320x320 Square": 5,
    "384x384 Square": 6,
    "448x448 Square": 7,
    "512x512 Square": 8,
    
    # Portrait (2:3)
    "128x192 Portrait (2:3)": 9,
    "256x384 Portrait (2:3)": 10,
    
    # Landscape (3:2)
    "192x128 Landscape (3:2)": 11,
    "384x256 Landscape (3:2)": 12
}

# Dictionary mapping from integer class to crop shape text
INT_TO_SHAPE = {v: k for k, v in SHAPE_TO_INT.items()}

# Dictionary mapping from integer class to dimensions (width, height)
INT_TO_DIMENSIONS = {
    # Squares (1:1)
    1: (64, 64),
    2: (128, 128),
    3: (192, 192),
    4: (256, 256),
    5: (320, 320),
    6: (384, 384),
    7: (448, 448),
    8: (512, 512),
    
    # Portrait (2:3)
    9: (128, 192),
    10: (256, 384),
    
    # Landscape (3:2)
    11: (192, 128),
    12: (384, 256)
}

def get_int_from_shape(shape_text):
    """Convert a crop shape text to its integer class value."""
    return SHAPE_TO_INT.get(shape_text, 4)  # Default to 256x256 Square (class 4) if not found

def get_shape_from_int(int_class):
    """Convert an integer class value to its crop shape text."""
    return INT_TO_SHAPE.get(int_class, "256x256 Square")  # Default to 256x256 Square if not found

def get_dimensions_from_int(int_class):
    """Get the width and height dimensions from an integer class value."""
    return INT_TO_DIMENSIONS.get(int_class, (256, 256))  # Default to 256x256 if not found

def get_num_classes():
    """Return the total number of shape classes."""
    return len(SHAPE_TO_INT)
