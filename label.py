import argparse
import csv
from flask import Flask, render_template_string, request, jsonify, send_from_directory
import os
import random
from PIL import Image
import torch
from model import ViTCropper

from utils import get_labels, save_labels
from predict import predict

app = Flask(__name__)
working_dir = ""
image_dir = ""
image_files = []
labels = {}

# HTML template embedded directly in the Python file
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IML Cropper GUI</title>
    <style>
        body {
            display: flex;
            flex-direction: column;
            align-items: center;
            font-family: Arial, sans-serif;
        }
        .controls {
            margin-top: 20px;
        }
        #image-container {
            position: relative;
            margin-top: 20px;
            width: 512px;
            height: 512px;
            background-color: #f0f0f0;
        }
        #image {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        #crop-square {
            position: absolute;
            border: 2px solid white;
            box-sizing: border-box;
            cursor: move;
            width: 256px;
            height: 256px;
            top: 128px;
            left: 128px;
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            grid-template-rows: 1fr 1fr 1fr;
        }
        .grid-line {
            position: absolute;
            background-color: white;
        }
        .grid-line-vertical {
            width: 1px;
            height: 100%;
            top: 0;
        }
        .grid-line-horizontal {
            width: 100%;
            height: 1px;
            left: 0;
        }
        .main-container {
            display: flex;
            justify-content: center;
            width: 100%;
        }
        .control-container {
            padding: 16px;
        }
    </style>
</head>
<body>
    <h1>IML Cropper GUI</h1>
    <div class="main-container">
        <div class="control-container">
            <div>
                <p>Image: <span id="image-index">0</span> / {{ total_images }}</p>
                <p>Filename: <span id="current-index"></span></p>
                <p>Total Labeled: <span id="total-labels">{{ total_labels }}</span></p>  
            </div>
            <div class="nav-container">
                <h3>Navigation</h3>
                <button onclick="prevImage()">Previous</button>
                <button onclick="nextImage()">Next</button>
                <button onclick="getRandomImage()">Random</button>
            </div>
            <div class="labeled-container">
                <h3>Labeled Images</h3>
                <label for="step">Step.</label>
                <br>
                <label><input type="radio" name="step" value="false" onclick="updateStep(false)"> NEG(-)</label>
                <label><input type="radio" name="step" value="true" checked onclick="updateStep(true)"> POS(+)</label>
                <button onclick="nextLabeledImage()">Next</button>
            </div>
            <div class="action-container">
                <h3>Action</h3>    
                <button onclick="saveLabel()">Save Label</button>
            </div>
            <div class="size-container">
                <label>Crop Size:</label>
                <div class="radio-group">
                    <label>
                        <input type="radio" name="crop-size" value="128"> 128px
                    </label>
                    <label>
                        <input type="radio" name="crop-size" value="192"> 192px
                    </label>
                    <label>
                        <input type="radio" name="crop-size" value="256" checked> 256px
                    </label>
                    <label>
                        <input type="radio" name="crop-size" value="320"> 320px
                    </label>
                    <label>
                        <input type="radio" name="crop-size" value="384"> 384px
                    </label>
                </div>
            </div>
            <div class="ratio-container">
                <label for="ratio">Ratio.</label>
                <br>
                <label><input type="radio" name="ratio" value="false" onclick="update_ratio(2,3)"> 2:3</label>
                <label><input type="radio" name="ratio" value="true" checked onclick="update_ratio(1,1)"> 1:1</label>
                <label><input type="radio" name="ratio" value="false" onclick="update_ratio(3,2)"> 3:2</label>
            </div>
        </div>
        <div id="image-container">
            <img id="image" src="" alt="Image to be labeled">
            <div id="crop-square">
                <!-- Vertical grid lines -->
                <div class="grid-line grid-line-vertical" style="left: 33.33%"></div>
                <div class="grid-line grid-line-vertical" style="left: 66.66%"></div>
                <!-- Horizontal grid lines -->
                <div class="grid-line grid-line-horizontal" style="top: 33.33%"></div>
                <div class="grid-line grid-line-horizontal" style="top: 66.66%"></div>
            </div>
        </div>
    </div>
    <script>
        let current_image = "";
        let next_labeled_image_step = true;
        let image_list = [];
        let labeled_images = [];
        let image_metadata = {}; // Store original image dimensions and padding info

        const total_images = parseInt("{{ total_images }}");
        let total_labels = parseInt("{{ total_labels }}");

        let crop_width = 256;
        let crop_height = 256;
        let crop_width_ratio = 1;
        let crop_height_ratio = 1;
        
        const crop_square = document.getElementById('crop-square');
        let isDragging = false;
        let startX, startY;

        document.getElementById('image').onload = function() {
            fetch(`/image_metadata/${current_image}`)
                .then(response => response.json())
                .then(data => {
                    image_metadata = data;
                    console.log('Image metadata:', image_metadata);
                    
                    if (labeled_images.includes(current_image)) {
                        console.log(`Loading label for ${current_image}`);
                        fetch(`/get_label/${current_image}`)
                            .then(response => response.json())
                            .then(label => {
                                if (label && label.coords) {
                                    const x1 = label.coords[0];
                                    const y1 = label.coords[1];
                                    const height = label.coords[2];
                                    const ratio_code = label.coords[3];
                                    
                                    const cropSize = Math.round(height * 512);
                                    document.querySelector(`input[name="crop-size"][value="${cropSize}"]`).checked = true;
                                    
                                    if (ratio_code === 0) {
                                        update_ratio(1, 1);
                                        document.querySelector('input[name="ratio"][value="true"]').checked = true;
                                    } else if (ratio_code === 1) {
                                        update_ratio(2, 3);
                                        document.querySelectorAll('input[name="ratio"]')[0].checked = true;
                                    } else if (ratio_code === 2) {
                                        update_ratio(3, 2);
                                        document.querySelectorAll('input[name="ratio"]')[2].checked = true;
                                    }
                                    
                                    applyCropWithRatio();
                                    
                                    crop_square.style.left = `${x1 * 512}px`;
                                    crop_square.style.top = `${y1 * 512}px`;
                                }
                            });
                    }
                })
                .catch(error => {
                    console.error('Error getting image metadata:', error);
                    applyCropWithRatio();
                });
        };
        
        function applyCropWithRatio() {
            const selectedSize = document.querySelector('input[name="crop-size"]:checked').value;
            const cropSize = parseInt(selectedSize);
            
            crop_width = cropSize;
            crop_height = crop_width * (crop_height_ratio / crop_width_ratio);
            
            const offsetX = (512 - crop_width) / 2;
            const offsetY = (512 - crop_height) / 2;
            
            drawCropSquare(offsetX, offsetY, offsetX + crop_width, offsetY + crop_height);
        }

        crop_square.addEventListener('mousedown', (e) => {
            isDragging = true;
            startX = e.clientX - crop_square.offsetLeft;
            startY = e.clientY - crop_square.offsetTop;
        });

        document.addEventListener('mousemove', (e) => {
            if (isDragging) {
                let x = e.clientX - startX;
                let y = e.clientY - startY;
                const containerRect = document.getElementById('image-container').getBoundingClientRect();
                const squareRect = crop_square.getBoundingClientRect();
                if (x < 0) x = 0;
                if (y < 0) y = 0;
                if (x + squareRect.width > containerRect.width) x = containerRect.width - squareRect.width;
                if (y + squareRect.height > containerRect.height) y = containerRect.height - squareRect.height;
                crop_square.style.left = x + 'px';
                crop_square.style.top = y + 'px';
            }
        });

        document.addEventListener('mouseup', () => {
            isDragging = false;
        });

        function resize_crop() {
            const selectedSize = document.querySelector('input[name="crop-size"]:checked').value;
            crop_width = parseInt(selectedSize);
            crop_height = crop_width * (crop_height_ratio / crop_width_ratio);
            
            const left = parseFloat(crop_square.style.left) || 128;
            const top = parseFloat(crop_square.style.top) || 128;
            
            crop_square.style.width = crop_width + 'px';
            crop_square.style.height = crop_height + 'px';
            
            const container = document.getElementById('image-container');
            if (left + crop_width > container.offsetWidth) {
                crop_square.style.left = (container.offsetWidth - crop_width) + 'px';
            }
            if (top + crop_height > container.offsetHeight) {
                crop_square.style.top = (container.offsetHeight - crop_height) + 'px';
            }
        }

        function update_ratio(width, height) {
            crop_width_ratio = width;
            crop_height_ratio = height;
            resize_crop();
        }

        function drawCropSquare(x1, y1, x2, y2) {
            crop_square.style.left = `${x1}px`;
            crop_square.style.top = `${y1}px`;
            crop_square.style.width = `${x2 - x1}px`;
            crop_square.style.height = `${y2 - y1}px`;
        }

        function loadRandomImage() {
            if (image_list.length === 0) return;
            const randomIndex = Math.floor(Math.random() * image_list.length);
            const randomImage = image_list[randomIndex];
            loadImage(randomImage);
        }

        window.onload = function() {
            fetch('/get_images')
                .then(response => response.json())
                .then(data => {
                    image_list = data.images;
                    if (image_list.length > 0) {
                        current_image = image_list[0];
                        loadImage(current_image);
                    }
                });
            
            fetch('/get_labeled_images')
                .then(response => response.json())
                .then(data => {
                    labeled_images = data.images;
                });
            
            // Updated initial crop square dimensions to match the default crop size (256px)
            drawCropSquare(128, 128, 384, 384);
            
            document.querySelectorAll('input[name="crop-size"]').forEach(radio => {
                radio.addEventListener('change', resize_crop);
            });
        };

        function loadImage(imageName) {
            const imageUrl = `/image/${imageName}`;
            const currentCropSize = document.querySelector('input[name="crop-size"]:checked').value;
            document.getElementById('image').src = imageUrl;
            document.getElementById('current-index').innerText = imageName;
            const currentIndex = image_list.indexOf(imageName) + 1;
            document.getElementById('image-index').innerText = currentIndex;
            current_image = imageName;
            document.querySelector(`input[name="crop-size"][value="${currentCropSize}"]`).checked = true;
        }

        function nextImage() {
            if (image_list.length === 0) return;
            const currentIndex = image_list.indexOf(current_image);
            const nextIndex = (currentIndex + 1) % image_list.length;
            loadImage(image_list[nextIndex]);
        }

        function prevImage() {
            if (image_list.length === 0) return;
            const currentIndex = image_list.indexOf(current_image);
            const prevIndex = (currentIndex - 1 + image_list.length) % image_list.length;
            loadImage(image_list[prevIndex]);
        }

        function nextLabeledImage() {
            fetch('/next_labeled_image', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    'name': current_image,
                    'step': next_labeled_image_step
                })
            }).then(response => response.json())
            .then(data => {
                if (data.img_name) {
                    loadImage(data.img_name);
                }
            }).catch(error => {
                console.error(error);
            });
        }

        function getRandomImage() {
            fetch('/random_image')
                .then(response => response.json())
                .then(data => {
                    if (data.img_name) {
                        loadImage(data.img_name);
                    }
                })
                .catch(error => {
                    console.error('Error getting random image:', error);
                });
        }

        function updateStep(step) {
            next_labeled_image_step = step;
        }

        function saveLabel() {
            console.log(`Saving label for ${current_image}`);
            if (!current_image) return;
            
            const cropBox = document.getElementById('crop-square');
            const left = parseInt(cropBox.style.left.replace('px', '')) || 0;
            const top = parseInt(cropBox.style.top.replace('px', '')) || 0;
            const width = parseInt(cropBox.style.width.replace('px', '')) || 0;
            const height = parseInt(cropBox.style.height.replace('px', '')) || 0;
            
            let ratio_code = 0;
            if (crop_width_ratio === 1 && crop_height_ratio === 1) {
                ratio_code = 0;
            } else if (crop_width_ratio === 2 && crop_height_ratio === 3) {
                ratio_code = 1;
            } else if (crop_width_ratio === 3 && crop_height_ratio === 2) {
                ratio_code = 2;
            }
            
            const x1_norm = left / 512;
            const y1_norm = top / 512;
            const height_norm = height / 512;
            
            fetch('/label', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    name: current_image,
                    x1: x1_norm,
                    y1: y1_norm,
                    height: height_norm,
                    ratio: ratio_code
                }),
            })
            .then(response => response.json())
            .then(data => {
                console.log('Label saved:', data);
                if (data.success) {
                    if (!labeled_images.includes(current_image)) {
                        labeled_images.push(current_image);
                        document.getElementById('total-labels').textContent = labeled_images.length;
                        console.log(`Added ${current_image} to labeled_images. Total: ${labeled_images.length}`);
                    }
                }
            })
            .catch(error => {
                console.error('Error saving label:', error);
            });
        }
    </script>
</body>
</html>
"""

def get_next_img_name(img_name: str, positive_step: bool, labels: dict[str, list[int]], all_images: list[str]) -> str:
    labeled_names = sorted(labels.keys())
    if not labeled_names:
        if img_name in all_images:
            current_index = all_images.index(img_name)
            if positive_step:
                next_index = (current_index + 1) % len(all_images)
            else:
                next_index = (current_index - 1) % len(all_images)
            return all_images[next_index]
        return img_name if img_name in all_images else all_images[0] if all_images else img_name
    
    if img_name in labeled_names:
        current_index = labeled_names.index(img_name)
        if positive_step:
            next_index = (current_index + 1) % len(labeled_names)
        else:
            next_index = (current_index - 1) % len(labeled_names)
        return labeled_names[next_index]
    
    for i, name in enumerate(labeled_names):
        if name > img_name:
            return labeled_names[i if positive_step else (i-1) % len(labeled_names)]
    
    return labeled_names[0 if positive_step else -1]

@app.route('/')
def index():
    global image_files, labels
    total_images = len(image_files)
    total_labels = len(labels)
    return render_template_string(HTML_TEMPLATE, total_images=total_images, total_labels=total_labels)

@app.route('/image/<path:img_name>')
def get_image(img_name):
    global image_dir
    if img_name in image_files:
        return send_from_directory(image_dir, img_name)
    return jsonify(error="Image not found"), 404

@app.route('/image_metadata/<path:img_name>')
def get_image_metadata(img_name):
    global image_dir
    if img_name not in image_files:
        return jsonify(error="Image not found"), 404
    
    try:
        img_path = os.path.join(image_dir, img_name)
        with Image.open(img_path) as img:
            orig_width, orig_height = img.size
            target_size = 224
            scale_factor = target_size / orig_height
            scaled_width = int(orig_width * scale_factor)
            padding_x = (target_size - scaled_width) // 2
            ui_scale = 512 / target_size
            return jsonify({
                "original_width": orig_width,
                "original_height": orig_height,
                "scaled_width": scaled_width,
                "padding_x": padding_x,
                "model_size": target_size,
                "ui_scale": ui_scale
            })
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/get_label/<path:img_name>')
def get_label(img_name):
    global labels
    if img_name in labels:
        return jsonify({"coords": labels[img_name]})
    return jsonify(error="Label not found"), 404

@app.route('/next_labeled_image', methods=['POST'])
def next_labeled_image():
    global labels, image_files
    data = request.json
    if data and "name" in data:
        next_img_name = get_next_img_name(
            img_name=data["name"], 
            positive_step=data["step"], 
            labels=labels,
            all_images=image_files
        )
        return jsonify(img_name=next_img_name)
    else:
        return jsonify(img_name=image_files[0] if image_files else "")

@app.route('/label', methods=['POST'])
def label_image():
    global labels, working_dir
    data = request.json
    if data and "name" in data:
        labels[data["name"]] = [data['x1'], data['y1'], data['height'], data['ratio']]
        save_labels(directory=working_dir, labels=labels)
        return jsonify(success=True)
    else:
        return jsonify(success=False)

@app.route('/get_images', methods=['GET'])
def get_images():
    global image_files
    return jsonify(images=image_files)

@app.route('/get_labeled_images', methods=['GET'])
def get_labeled_images():
    global labels
    return jsonify(images=list(labels.keys()))

@app.route('/random_image', methods=['GET'])
def random_image():
    global image_files
    if image_files:
        return jsonify(img_name=random.choice(image_files))
    return jsonify(img_name="")

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    global working_dir
    data = request.json
    img_name = data.get('id', None)
    
    if img_name is None:
        return jsonify({'error': 'No image name provided'}), 400
    
    img_path = os.path.join(image_dir, img_name)
    if not os.path.exists(img_path):
        return jsonify({'error': f'Image {img_name} not found'}), 404
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ViTCropper()
        model_path = os.path.join(working_dir, 'crop_model.pth')
        
        if not os.path.exists(model_path):
            return jsonify({'error': 'No trained model found. Please train the model first.'}), 404
            
        model.load_state_dict(torch.load(model_path, map_location=device))
        x1, y1, x2, y2 = predict(device, model, img_path)
        return jsonify({
            'prediction': {
                'x1': float(x1),
                'y1': float(y1),
                'x2': float(x2),
                'y2': float(y2)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def label(working_directory: str):
    global working_dir, image_files, image_dir, labels
    working_dir = working_directory
    image_dir = os.path.join(working_dir, 'src_culled')
    
    if not os.path.exists(image_dir):
        print(f"Error: src_culled directory not found at {image_dir}")
        return
    
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    labels = get_labels(working_dir)
    
    print(f"Starting labeling app with {len(image_files)} images and {len(labels)} existing labels.")
    print(f"Access the labeling interface at http://localhost:5000")
    
    src_cropped_dir = os.path.join(working_dir, 'src_cropped')
    os.makedirs(src_cropped_dir, exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the image labeling Flask app.")
    parser.add_argument("working_dir", type=str, help="Directory where the album is located (contains src_culled directory).")
    args = parser.parse_args()
    label(args.working_dir)
