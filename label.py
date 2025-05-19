import argparse
import csv
from flask import Flask, render_template_string, request, jsonify, send_from_directory
import os
import random
import size_conversion  # Import the new size conversion module

app = Flask(__name__)
image_dir = ""
image_files = []
labels = {}
labels_file = ""
current_image_index = -1  # Index for image_files list

# HTML template updated per spec
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <title>IML Cropper GUI</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; display: flex; flex-direction: column; min-height: 100vh; background-color: white; color: black; }
        .header-section { padding: 20px; text-align: center; }
        .header-section h1 { margin: 0; font-size: 2.5em; }
        .header-section p { margin: 5px 0 0; font-size: 1.2em; color: black; }
        .main-content { display: flex; flex: 1; padding: 20px; }
        .left-panel { flex: 0 0 550px; /* Fixed width for image panel */ margin-right: 20px; }
        .right-panel { flex: 1; display: flex; flex-direction: column; }
        section { margin-bottom: 20px; padding:15px; }
        .panel-section { margin-bottom: 20px; padding: 15px; }
        h2 { padding-bottom: 8px; margin-top:0; }
        select, button { margin: 4px; padding: 8px; }
        #image-container { width: 512px; height: 512px; background: white; position: relative; border: 1px solid black; }
        #image { width: 100%; height: 100%; object-fit: contain; }
        #crop-square { position: absolute; border: 2px solid red; cursor: move; display: grid; grid-template: repeat(3,1fr)/repeat(3,1fr); }
        .grid-line-vertical { position: absolute; width: 1px; height: 100%; background: rgba(255,0,0,0.3); }
        .grid-line-horizontal { position: absolute; height: 1px; width: 100%; background: rgba(255,0,0,0.3); }
        ul { padding-left: 20px; }
        /* Radio button styles - stacked as a list */
        #crop-shape-radio-group { display: block; }
        .radio-option { margin: 5px 0; display: block; }
        .radio-option input[type="radio"] { margin-right: 5px; }
        .radio-option label { cursor: pointer; }
        .shape-stats { color: #666; font-size: 0.9em; margin-left: 5px; }
        .category-heading { margin-top: 12px; margin-bottom: 6px; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header-section">
        <h1>IML Crop</h1>
        <p>royokello</p>
    </div>
    <div class="main-content">
        <div class="left-panel">
            <section id="image-section">
                <h2>Image</h2>
                <p id="image-counter">Loading...</p>
                <div id="image-container">
                    <img id="image" src="" alt="Image to be labeled" />
                    <div id="crop-square">
                        <div class="grid-line-vertical" style="left:33.33%"></div>
                        <div class="grid-line-vertical" style="left:66.66%"></div>
                        <div class="grid-line-horizontal" style="top:33.33%"></div>
                        <div class="grid-line-horizontal" style="top:66.66%"></div>
                    </div>
                </div>
            </section>
        </div>
        <div class="right-panel">
            <h2>Control</h2>
            <div class="panel-section">
                <button id="nav-prev-btn">Previous Image</button>
                <button id="nav-next-btn">Next Image</button>
                <button id="nav-random-btn">Random Image</button>
            </div>
            
            <div class="panel-section">
                <button id="prev-btn">Previous Label</button>
                <button id="next-btn">Next Label</button>
                <button id="save-btn">Save</button>
                <button id="delete-btn">Delete</button>
            </div>
            
            <div class="panel-section" id="crop-shape-section">
                <div id="crop-shape-radio-group"></div>
            </div>
        </div>
    </div>

    <script>
        let imageList = [];
        let labeledList = [];
        let currentImage = null;

        const CROP_SHAPES = [
            // Squares (1:1)
            { text: "64x64 Square", width: 64, height: 64 },
            { text: "128x128 Square", width: 128, height: 128 },
            { text: "192x192 Square", width: 192, height: 192 },
            { text: "256x256 Square", width: 256, height: 256, default: true },
            { text: "320x320 Square", width: 320, height: 320 },
            { text: "384x384 Square", width: 384, height: 384 },
            { text: "448x448 Square", width: 448, height: 448 },
            { text: "512x512 Square", width: 512, height: 512 },
            // Portrait (2:3)
            { text: "128x192 Portrait (2:3)", width: 128, height: 192 },
            { text: "256x384 Portrait (2:3)", width: 256, height: 384 },
            // Landscape (3:2)
            { text: "192x128 Landscape (3:2)", width: 192, height: 128 },
            { text: "384x256 Landscape (3:2)", width: 384, height: 256 }
        ];

        // Function to fetch and update shape statistics
        function fetchShapeStats() {
            fetch('/shape_stats')
                .then(response => response.json())
                .then(data => {
                    // Create a map of shape_int to count
                    const countMap = {};
                    data.stats.forEach(stat => {
                        countMap[stat.shape_int] = stat.count;
                    });
                    
                    // Update the stats display for each radio button
                    CROP_SHAPES.forEach(shape => {
                        const shapeInt = getShapeIntFromText(shape.text);
                        const count = countMap[shapeInt] || 0;
                        const statsSpan = document.getElementById(`stats-${shapeInt}`);
                        if (statsSpan) {
                            statsSpan.textContent = `(${count})`;
                        }
                    });
                })
                .catch(error => console.error("Error fetching shape stats:", error));
        }
        
        // Helper function to get shape_int from shape text
        function getShapeIntFromText(shapeText) {
            // This should match the logic in size_conversion.py
            const shapeMap = {
                "64x64 Square": 1,
                "128x128 Square": 2,
                "192x192 Square": 3,
                "256x256 Square": 4,
                "320x320 Square": 5,
                "384x384 Square": 6,
                "448x448 Square": 7,
                "512x512 Square": 8,
                "128x192 Portrait (2:3)": 9,
                "256x384 Portrait (2:3)": 10,
                "192x128 Landscape (3:2)": 11,
                "384x256 Landscape (3:2)": 12
            };
            return shapeMap[shapeText] || 4; // Default to 4 (256x256 Square)
        }

        function populateCropShapeRadioButtons() {
            const container = document.getElementById('crop-shape-radio-group');
            let defaultShapeText = '';
            
            // Group shapes by category
            const categories = {
                "Squares (1:1)": [],
                "Portrait (2:3)": [],
                "Landscape (3:2)": []
            };
            
            // Sort shapes into categories
            CROP_SHAPES.forEach(shape => {
                if (shape.text.includes("Square")) {
                    categories["Squares (1:1)"].push(shape);
                } else if (shape.text.includes("Portrait")) {
                    categories["Portrait (2:3)"].push(shape);
                } else if (shape.text.includes("Landscape")) {
                    categories["Landscape (3:2)"].push(shape);
                }
                
                if (shape.default) {
                    defaultShapeText = shape.text;
                }
            });
            
            // Create radio buttons by category
            for (const [category, shapes] of Object.entries(categories)) {
                // Add category heading
                const categoryHeading = document.createElement('div');
                categoryHeading.className = 'category-heading';
                categoryHeading.textContent = category;
                container.appendChild(categoryHeading);
                
                // Add radio buttons for this category
                shapes.forEach(shape => {
                    const shapeInt = getShapeIntFromText(shape.text);
                    const wrapper = document.createElement('div');
                    wrapper.className = 'radio-option';
                    
                    const radio = document.createElement('input');
                    radio.type = 'radio';
                    radio.name = 'crop-shape';
                    radio.id = `shape-${shapeInt}`;
                    radio.value = shape.text;
                    radio.checked = shape.text === defaultShapeText;
                    radio.addEventListener('change', applyCrop);
                    
                    const label = document.createElement('label');
                    label.htmlFor = `shape-${shapeInt}`;
                    label.textContent = shape.text;
                    
                    const statsSpan = document.createElement('span');
                    statsSpan.className = 'shape-stats';
                    statsSpan.id = `stats-${shapeInt}`;
                    statsSpan.textContent = '(0)'; // Initial value, will be updated by fetchShapeStats
                    
                    wrapper.appendChild(radio);
                    wrapper.appendChild(label);
                    wrapper.appendChild(statsSpan);
                    container.appendChild(wrapper);
                });
            }
            
            // Fetch initial stats
            fetchShapeStats();
        }

        function navigatePrev() {
            fetch('/navigate?action=prev')
                .then(response => response.json())
                .then(data => {
                    if (data.img_name) {
                        loadImage(data.img_name);
                        if (data.index !== undefined) {
                            currentImageIndex = data.index;
                            updateImageCounter();
                        }
                    }
                    else if (data.error) console.error("Error navigating prev:", data.error);
                });
        }

        function navigateNext() {
            fetch('/navigate?action=next')
                .then(response => response.json())
                .then(data => {
                    if (data.img_name) {
                        loadImage(data.img_name);
                        if (data.index !== undefined) {
                            currentImageIndex = data.index;
                            updateImageCounter();
                        }
                    }
                    else if (data.error) console.error("Error navigating next:", data.error);
                });
        }

        function navigateRandom() {
            fetch('/navigate?action=random')
                .then(response => response.json())
                .then(data => {
                    if (data.img_name) {
                        loadImage(data.img_name);
                        if (data.index !== undefined) {
                            currentImageIndex = data.index;
                            updateImageCounter();
                        }
                    }
                    else if (data.error) console.error("Error navigating random:", data.error);
                });
        }

        // Global variables to track image counts
        let totalImages = 0;
        let totalLabels = 0;
        let currentImageIndex = 0;

        function updateImageCounter() {
            // Format: "1/2000 (100 labeled)"
            const counterText = `${currentImageIndex + 1}/${totalImages} (${totalLabels} labeled)`;
            document.getElementById('image-counter').innerText = counterText;
        }

        function fetchStats() {
            fetch('/stats').then(r=>r.json()).then(data=>{
                totalImages = data.total_images;
                totalLabels = data.total_labels;
                updateImageCounter();
            });
        }

        function fetchSizes() {
            fetch('/sizes').then(r=>r.json()).then(data=>{
                document.getElementById('unique-sizes').innerText = data.unique_sizes;
                const ul = document.getElementById('size-stats'); ul.innerHTML = '';
                data.stats.forEach(s=>{
                    const li = document.createElement('li');
                    li.innerText = `${s.size}px: ${s.count}`;
                    ul.appendChild(li);
                });
            });
        }

        function fetchRatios() {
            fetch('/ratios').then(r=>r.json()).then(data=>{
                document.getElementById('unique-ratios').innerText = data.unique_ratios;
                const ul = document.getElementById('ratio-stats'); ul.innerHTML = '';
                data.stats.forEach(r=>{
                    const li = document.createElement('li');
                    li.innerText = `${r.ratio}: ${r.count}`;
                    ul.appendChild(li);
                });
            });
        }

        function applyCrop() {
            const sq = document.getElementById('crop-square');
            
            // Get the selected radio button
            const selectedRadio = document.querySelector('input[name="crop-shape"]:checked');
            
            // Return early if no radio button is selected
            if (!selectedRadio) {
                console.warn('applyCrop called before any radio button is selected');
                return; 
            }
            
            const selectedShapeText = selectedRadio.value;
            const selectedShape = CROP_SHAPES.find(shape => shape.text === selectedShapeText);

            if (!selectedShape) {
                console.error('Selected shape not found:', selectedShapeText);
                return;
            }

            const imgContainer = document.getElementById('image-container');
            const containerWidth = imgContainer.clientWidth;
            const containerHeight = imgContainer.clientHeight;

            // Store current position. Initial position is set by populateCropShapeDropdown.
            let currentLeft = parseInt(sq.style.left) || 0;
            let currentTop = parseInt(sq.style.top) || 0;

            let targetWidthForCalc = selectedShape.width;
            let targetHeightForCalc = selectedShape.height;

            let scaleX = 1.0;
            let scaleY = 1.0;

            if (targetWidthForCalc > 0 && targetWidthForCalc > containerWidth) {
                scaleX = containerWidth / targetWidthForCalc;
            }
            if (targetHeightForCalc > 0 && targetHeightForCalc > containerHeight) {
                scaleY = containerHeight / targetHeightForCalc;
            }
            
            let actualScale = Math.min(scaleX, scaleY);
            actualScale = Math.min(actualScale, 1.0); // Don't upscale if fits

            const finalNewWidth = targetWidthForCalc * actualScale;
            const finalNewHeight = targetHeightForCalc * actualScale;

            sq.style.width = finalNewWidth + 'px';
            sq.style.height = finalNewHeight + 'px';

            const maxLeft = Math.max(0, containerWidth - finalNewWidth); 
            const maxTop = Math.max(0, containerHeight - finalNewHeight);

            sq.style.left = Math.max(0, Math.min(currentLeft, maxLeft)) + 'px';
            sq.style.top = Math.max(0, Math.min(currentTop, maxTop)) + 'px';
        }

        function makeDraggable() {
            const cropSquare = document.getElementById('crop-square');
            const imageContainer = document.getElementById('image-container');
            let offsetX, offsetY, isDragging = false;

            cropSquare.addEventListener('mousedown', (e) => {
                isDragging = true;
                // Calculate offset from top-left of cropSquare to mouse click position
                offsetX = e.clientX - cropSquare.getBoundingClientRect().left;
                offsetY = e.clientY - cropSquare.getBoundingClientRect().top;
                cropSquare.style.cursor = 'grabbing'; // Change cursor while dragging

                // Prevent text selection during drag
                e.preventDefault(); 
            });

            document.addEventListener('mousemove', (e) => {
                if (!isDragging) return;

                // Calculate new position relative to imageContainer
                // 1. Get mouse position relative to the viewport
                let mouseX = e.clientX;
                let mouseY = e.clientY;

                // 2. Get imageContainer's position relative to the viewport
                const containerRect = imageContainer.getBoundingClientRect();

                // 3. Calculate desired top-left of cropSquare relative to imageContainer's top-left
                let newLeft = mouseX - containerRect.left - offsetX;
                let newTop = mouseY - containerRect.top - offsetY;

                // 4. Constrain to imageContainer boundaries
                const maxLeft = imageContainer.clientWidth - cropSquare.offsetWidth;
                const maxTop = imageContainer.clientHeight - cropSquare.offsetHeight;

                newLeft = Math.max(0, Math.min(newLeft, maxLeft));
                newTop = Math.max(0, Math.min(newTop, maxTop));

                cropSquare.style.left = newLeft + 'px';
                cropSquare.style.top = newTop + 'px';
            });

            document.addEventListener('mouseup', () => {
                if (isDragging) {
                    isDragging = false;
                    cropSquare.style.cursor = 'move'; // Reset cursor
                }
            });
        }

        function init() {
            // Fetch the full image list (might be used by other parts like labeled navigation)
            fetch('/get_images').then(r=>r.json()).then(d=>{ imageList = d.images; });

            // Load initial image using the new server-side indexed navigation
            fetch('/navigate?action=load_initial')
                .then(response => response.json())
                .then(data => {
                    if (data.img_name) {
                        loadImage(data.img_name);
                        // Enable nav buttons if they were potentially disabled
                        if(document.getElementById('nav-prev-btn')) document.getElementById('nav-prev-btn').disabled = false;
                        if(document.getElementById('nav-next-btn')) document.getElementById('nav-next-btn').disabled = false;
                        if(document.getElementById('nav-random-btn')) document.getElementById('nav-random-btn').disabled = false;
                    } else {
                        console.error("Error loading initial image:", data.error || "No image name returned");
                        // Handle no images available: display a message, disable buttons etc.
                        if(document.getElementById('nav-prev-btn')) document.getElementById('nav-prev-btn').disabled = true;
                        if(document.getElementById('nav-next-btn')) document.getElementById('nav-next-btn').disabled = true;
                        if(document.getElementById('nav-random-btn')) document.getElementById('nav-random-btn').disabled = true;
                    }
                });
            fetch('/get_labeled_images').then(r=>r.json()).then(d=>{
                labeledList = d.images;
                totalLabels = labeledList.length; // Update totalLabels when labeled images are fetched
                updateImageCounter();
            });
            fetchStats(); // Get total images and labels
            populateCropShapeRadioButtons(); // Set up crop shape radio buttons

            document.getElementById('prev-btn').addEventListener('click', ()=>navigateLabel(false));
            document.getElementById('next-btn').addEventListener('click', ()=>navigateLabel(true));
            document.getElementById('save-btn').addEventListener('click', saveImage);
            document.getElementById('delete-btn').onclick = deleteImage; // Corrected to call existing deleteImage function

            // Event listeners for NEW navigation buttons
            if(document.getElementById('nav-prev-btn')) document.getElementById('nav-prev-btn').onclick = navigatePrev;
            if(document.getElementById('nav-next-btn')) document.getElementById('nav-next-btn').onclick = navigateNext;
            if(document.getElementById('nav-random-btn')) document.getElementById('nav-random-btn').onclick = navigateRandom;

            makeDraggable(); // Initialize drag functionality
        }

        let initialSetupDone = false; // Flag to ensure initial setup runs once

        function onImageLoadAndReady() {
            if (!initialSetupDone) {
                applyCrop(); // Apply crop for the initially selected shape

                // Set specific default position to (128, 128)
                const cropSquareElement = document.getElementById('crop-square');
                const imageContainerElement = document.getElementById('image-container');
                
                // Check if container has rendered and has dimensions
                if (cropSquareElement && imageContainerElement && imageContainerElement.clientWidth > 0 && imageContainerElement.clientHeight > 0) {
                    let defaultLeft = 128;
                    let defaultTop = 128;

                    // Ensure the default position doesn't push the box out of bounds
                    const maxLeft = imageContainerElement.clientWidth - cropSquareElement.offsetWidth;
                    const maxTop = imageContainerElement.clientHeight - cropSquareElement.offsetHeight;

                    cropSquareElement.style.left = Math.max(0, Math.min(defaultLeft, maxLeft)) + 'px';
                    cropSquareElement.style.top = Math.max(0, Math.min(defaultTop, maxTop)) + 'px';
                    
                    initialSetupDone = true; // Mark initial setup as complete
                } 
            }
        }

        // Function to load crop shape from label data
        function loadCropShapeFromLabel(imageName) {
            // Check if the image has a label
            fetch(`/get_label?name=${encodeURIComponent(imageName)}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success && data.label) {
                        const label = data.label;
                        const cropShapeInt = label[2]; // The third element is the crop shape integer
                        const cropShapeText = data.shape_text; // Server will provide the text equivalent
                        
                        // Set the radio button to the correct shape
                        const radioId = `shape-${cropShapeInt}`;
                        const radioButton = document.getElementById(radioId);
                        if (radioButton) {
                            radioButton.checked = true;
                        }
                        
                        // Apply the crop with the loaded dimensions
                        applyCrop();
                        
                        // Set the crop box position based on the label
                        const sq = document.getElementById('crop-square');
                        const imgContainer = document.getElementById('image-container');
                        if (sq && imgContainer) {
                            const containerWidth = imgContainer.clientWidth;
                            const containerHeight = imgContainer.clientHeight;
                            
                            // Convert normalized coordinates to pixels
                            const left = label[0] * containerWidth;
                            const top = label[1] * containerHeight;
                            
                            sq.style.left = `${left}px`;
                            sq.style.top = `${top}px`;
                        }
                    }
                })
                .catch(error => console.error("Error loading label:", error));
        }

        function loadImage(name) {
            currentImage = name;
            const imgElement = document.getElementById('image');
            
            // Set up the onload event handler
            imgElement.onload = () => {
                onImageLoadAndReady(); // Call our new function when image is actually loaded
                updateImageCounter(); // Update the counter when image loads
                
                // Check if this image has a label and load its crop shape
                if (initialSetupDone) { // Only try to load label after initial setup
                    loadCropShapeFromLabel(name);
                }
            };
            imgElement.onerror = () => {
                console.error("Error loading image:", name);
                document.getElementById('image-counter').innerText = "Error loading image";
            };
            
            imgElement.src = `/image/${name}`;
        }

        function navigateLabel(step) {
            fetch('/next_labeled_image', {
                method: 'POST', headers: {'Content-Type':'application/json'},
                body: JSON.stringify({name:currentImage, step})
            }).then(r=>r.json()).then(d=>loadImage(d.img_name));
        }

        function saveImage() {
            const sq = document.getElementById('crop-square');
            const imgContainer = document.getElementById('image-container');
            
            // Ensure container dimensions are available, fallback to 512 if not (e.g. not fully rendered)
            const containerWidth = imgContainer.clientWidth > 0 ? imgContainer.clientWidth : 512;
            const containerHeight = imgContainer.clientHeight > 0 ? imgContainer.clientHeight : 512;

            const left = parseFloat(sq.style.left) / containerWidth;
            const top = parseFloat(sq.style.top) / containerHeight;

            // Get the selected radio button
            const selectedRadio = document.querySelector('input[name="crop-shape"]:checked');
            if (!selectedRadio) {
                console.error('No crop shape selected');
                return;
            }
            const selectedShapeText = selectedRadio.value;

            // We still send the text to the server, which will convert it to an integer
            // This keeps the client-side code simpler and more maintainable
            fetch('/label', {
                method:'POST', headers:{'Content-Type':'application/json'},
                body:JSON.stringify({name:currentImage, x:left, y:top, crop_shape:selectedShapeText})
            }).then(r=>r.json()).then(()=>{
                // Update label count and refresh counter
                totalLabels++;
                updateImageCounter();
                // Fetch updated stats
                fetchStats();
                // Update shape stats
                fetchShapeStats();
            });
        }

        function deleteImage() {
            fetch('/delete_label', {
                method:'POST', headers:{'Content-Type':'application/json'},
                body:JSON.stringify({name:currentImage})
            }).then(r=>r.json()).then(()=>{
                // Update label count and refresh counter
                totalLabels = Math.max(0, totalLabels - 1);
                updateImageCounter();
                // Fetch updated stats
                fetchStats();
                // Update shape stats
                fetchShapeStats();
            });
        }

        window.onload = init;
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/stats')
def stats():
    return jsonify(total_images=len(image_files), total_labels=len(labels))

@app.route('/shape_stats')
def shape_stats():
    # Count labels by crop shape integer
    shape_counts = {}
    for label_data in labels.values():
        shape_int = label_data[2]  # The third element is the crop shape integer
        shape_counts[shape_int] = shape_counts.get(shape_int, 0) + 1
    
    # Convert to a list of objects with shape text and count
    stats = []
    for shape_int, count in shape_counts.items():
        shape_text = size_conversion.get_shape_from_int(shape_int)
        dimensions = size_conversion.get_dimensions_from_int(shape_int)
        stats.append({
            'shape_int': shape_int,
            'shape_text': shape_text,
            'width': dimensions[0],
            'height': dimensions[1],
            'count': count
        })
    
    # Sort by shape_int for consistent ordering
    stats.sort(key=lambda x: x['shape_int'])
    
    return jsonify(stats=stats)

@app.route('/sizes')
def sizes():
    counts = {}
    for v in labels.values():
        size = round(v[2] * 512)
        counts[size] = counts.get(size, 0) + 1
    return jsonify(
        unique_sizes=len(counts),
        stats=[{'size': k, 'count': counts[k]} for k in sorted(counts)]
    )

@app.route('/ratios')
def ratios():
    counts = {}
    for v in labels.values():
        rat = round(v[3], 4)
        counts[rat] = counts.get(rat, 0) + 1
    return jsonify(
        unique_ratios=len(counts),
        stats=[{'ratio': k, 'count': counts[k]} for k in sorted(counts)]
    )

@app.route('/delete_label', methods=['POST'])
def delete_label():
    global labels_file # Ensure we are using the global one correctly
    data = request.json
    name = data.get('name')
    if name and name in labels:
        labels.pop(name)
        try:
            # Read existing content, filter out the deleted label
            with open(labels_file, 'r', newline='') as f_read:
                reader = csv.reader(f_read)
                header = next(reader) # Assuming there's always a header
                lines_after_delete = [row for row in reader if row[0] != name]
            
            # Write back the header and the filtered lines
            with open(labels_file, 'w', newline='') as f_write:
                writer = csv.writer(f_write)
                writer.writerow(header)
                writer.writerows(lines_after_delete)
            return jsonify(success=True)
        except Exception as e:
            print(f"Error processing CSV file during delete: {e}")
            # Optionally, re-add label to memory if CSV write fails to maintain consistency?
            # For now, just report error.
            return jsonify(success=False, error=str(e))

    return jsonify(success=False, error="Label not found or name not provided")

@app.route('/image/<path:img_name>')
def get_image(img_name):
    if img_name in image_files:
        return send_from_directory(image_dir, img_name)
    return jsonify(error='Image not found'), 404

@app.route('/get_images')
def get_images():
    return jsonify(images=image_files)

@app.route('/get_labeled_images')
def get_labeled_images():
    return jsonify(images=list(labels.keys()))

@app.route('/get_label')
def get_label():
    name = request.args.get('name')
    if name and name in labels:
        label_data = labels[name]
        # Convert the integer class back to text for the client
        crop_shape_int = label_data[2]
        shape_text = size_conversion.get_shape_from_int(crop_shape_int)
        return jsonify(success=True, label=label_data, shape_text=shape_text)
    return jsonify(success=False, error="Label not found")

@app.route('/next_labeled_image', methods=['POST'])
def next_labeled_image():
    data = request.json
    name = data.get('name', image_files[0])
    img_name = get_next_img_name(name, data.get('step', True), labels, image_files)
    return jsonify(img_name=img_name)

@app.route('/label', methods=['POST'])
def label_image():
    data = request.json
    name = data.get('name')
    if name:
        x = data['x']
        y = data['y']
        crop_shape_text = data['crop_shape'] # Text from JS
        
        # Convert crop shape text to integer class
        crop_shape = size_conversion.get_int_from_shape(crop_shape_text)
        
        # Store the integer class in memory
        labels[name] = [x, y, crop_shape]
        
        # Check if file exists to write header, or append
        file_exists = os.path.isfile(labels_file)
        
        with open(labels_file, 'a', newline='') as f:
            writer = csv.writer(f)
            # Write header if file is new or empty
            if not file_exists or os.path.getsize(labels_file) == 0:
                writer.writerow(["name", "x", "y", "crop_shape"]) # Updated header
            writer.writerow([name, x, y, crop_shape])
        
        return jsonify(success=True)
    return jsonify(success=False)

@app.route('/navigate')
def navigate_images():
    global current_image_index, image_files
    action = request.args.get('action')

    if not image_files:
        return jsonify(error="No images available", img_name=None, index=-1)

    num_images = len(image_files)
    if action == 'load_initial':
        current_image_index = 0
    elif action == 'next':
        current_image_index = (current_image_index + 1) % num_images
    elif action == 'prev':
        current_image_index = (current_image_index - 1 + num_images) % num_images
    elif action == 'random':
        current_image_index = random.randint(0, num_images - 1)
    # If current_image_index was -1 (e.g. from initial state with no images, then images loaded)
    # and action is not load_initial, ensure it becomes valid.
    elif current_image_index == -1 and num_images > 0:
        current_image_index = 0 # Default to first image
    elif not (0 <= current_image_index < num_images):
         # if somehow index is out of bounds and not handled above, reset to 0
        current_image_index = 0

    if 0 <= current_image_index < num_images:
        return jsonify(img_name=image_files[current_image_index], index=current_image_index)
    else:
        # This case should ideally not be reached if image_files is populated
        return jsonify(error="Image index out of bounds after action", img_name=None, index=current_image_index)

def find_latest_stage(project: str) -> int:
    """Find the latest stage in the project directory.
    
    Args:
        project: Path to the project directory
        
    Returns:
        The highest stage number found
        
    Raises:
        ValueError: If no stage directories are found
    """
    stage_dirs = []
    for item in os.listdir(project):
        if os.path.isdir(os.path.join(project, item)) and item.startswith('stage_'):
            try:
                stage_num = int(item.split('_')[1])
                stage_dirs.append(stage_num)
            except (IndexError, ValueError):
                continue
    
    if stage_dirs:
        return max(stage_dirs)
    else:
        raise ValueError(f"No stage directories found in {project}. Create at least one stage directory (e.g., 'stage_1').")

def main(project: str, stage: int = None):
    global image_dir, image_files, labels, labels_file, current_image_index
    
    # If no stage provided, find the latest one
    if stage is None:
        try:
            stage = find_latest_stage(project)
            print(f"Using latest stage: {stage}")
        except ValueError as e:
            print(f"Error: {e}")
            return
    
    # Form the directory name from the stage number
    stage_dir = f"stage_{stage}"
    
    # Check if the directory exists
    image_dir = os.path.join(project, stage_dir)
    if not os.path.exists(image_dir):
        print(f"Error: Stage directory 'stage_{stage}' not found in {project}")
        return
    
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png','.jpg','.jpeg','.gif','.bmp'))]
    if image_files:
        current_image_index = 0
    else:
        current_image_index = -1
    
    labels = {}
    labels_file = os.path.join(project, f'stage_{stage}_crop_labels.csv')

    if os.path.exists(labels_file):
        with open(labels_file, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None) # Read header (and skip it for data processing)
            
            # For robustness, check if header matches expected format
            expected_headers = [["name", "x", "y", "crop_shape"], ["name", "x", "y", "crop_shape_int"]]
            header_is_valid = header is None or any(header == expected for expected in expected_headers)
            
            if not header_is_valid and header is not None:
                print(f"Warning: CSV header is {header}, expected one of {expected_headers}. "
                      "This might cause issues if the file is from an incompatible version.")
            
            for row in reader:
                if len(row) == 4: # Expects name, x, y, crop_shape/crop_shape_int (4 columns)
                    img_name = row[0]
                    try:
                        x = float(row[1])
                        y = float(row[2])
                        
                        # Handle both string and integer formats for backward compatibility
                        if row[3].isdigit():
                            # If it's a digit string, convert to int
                            crop_shape = int(row[3])
                        else:
                            # If it's a text description, convert to int using the mapping
                            crop_shape = size_conversion.get_int_from_shape(row[3])
                            
                        labels[img_name] = [x, y, crop_shape]
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse row for {img_name}: {row}. Error: {e}. Skipping.")
                elif row: # If row is not empty but has wrong number of columns
                    print(f"Warning: Malformed row in CSV for {row[0] if row else 'Unknown Image'}: {row}. Expected 4 columns. Skipping.")
    
    print(f"Starting app with {len(image_files)} images and {len(labels)} labels.")
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Launch web interface for labeling images.")
    parser.add_argument('--project', type=str, required=True,
                        help='Root project directory containing image directories')
    parser.add_argument('--stage', type=int, default=None,
                        help='Stage number to label (e.g., 1, 2, etc.). If not provided, uses the latest stage.')
    args = parser.parse_args()
    main(args.project, args.stage)
