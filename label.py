import argparse
import csv
import json
import os
import random
from flask import Flask, render_template, request, jsonify, send_from_directory, make_response
from utils import find_latest_stage

app = Flask(__name__)

# Globals
image_dir = ""
image_files = []
# Keyed by image index instead of filename
labels = {}         # { image_idx: [x, y, width, ratio_id], … }
labels_file = ""
# Ratios loaded from JSON to preserve continuous indices
ratios = []
ratios_file = ""
current_image_index = -1
labelled_idxs = []  # list of image indexes that have labels


def _get_ratio_stats(labels_dict):
    """
    Count occurrences of each fixed ratio string.
    Uses a hardcoded list to ensure stats keys are always present.
    """
    # Hardcoded ratio keys
    ratio_keys = ["1/1", "1/2", "1/3", "2/1", "2/3", "3/2"]
    stats = {r: 0 for r in ratio_keys}
    # Tally labels by their stored ratio_id
    for _, (_, _, _, rid) in labels_dict.items():
        if 0 <= rid < len(ratios):
            key = ratios[rid]
            if key in stats:
                stats[key] += 1
    return stats


@app.route('/')
def index():
    return render_template(
        "index.html",
        total_images=len(image_files),
        total_labels=len(labels),
        ratio_stats=_get_ratio_stats(labels),
        current_index=current_image_index
    )


@app.route('/image/<int:index>')
def get_image(index):
    try:
        img_name = image_files[index]
    except (IndexError, TypeError):
        return jsonify(error='Image not found'), 404

    # Prepare label header if exists
    label_data = None
    if index in labels:
        x, y, w, rid = labels[index]
        ratio_str = ratios[rid] if 0 <= rid < len(ratios) else None
        label_data = {'x': x, 'y': y, 'width': w, 'ratio': ratio_str}

    # Serve image and attach label JSON in header
    response = make_response(send_from_directory(image_dir, img_name))
    if label_data:
        response.headers['X-Label'] = json.dumps(label_data)
    return response


@app.route('/label', methods=['POST'])
def label_image():
    global labels, labelled_idxs
    data = request.json
    idx = data.get('index')
    is_delete = data.get('is_delete', False)

    # Validate index
    if not isinstance(idx, int) or idx < 0 or idx >= len(image_files):
        return jsonify(error='Invalid image index'), 400

    # Delete label
    if is_delete:
        if idx in labels:
            labels.pop(idx)
            if idx in labelled_idxs:
                labelled_idxs.remove(idx)
        # Rewrite CSV from scratch with header
        with open(labels_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'x', 'y', 'width', 'ratio'])
            for img_idx, (x, y, w, rid) in labels.items():
                writer.writerow([img_idx, x, y, w, rid])
        return jsonify(
            total_labels=len(labels),
            ratio_stats=_get_ratio_stats(labels)
        )

    # Add/update label
    try:
        x = float(data['x'])
        y = float(data['y'])
        w = float(data['width'])
        ratio_str = data['ratio']
    except (TypeError, ValueError):
        return jsonify(error='Invalid label data'), 400

    # Ensure ratio exists in loaded list
    if ratio_str not in ratios:
        ratios.append(ratio_str)
        with open(ratios_file, 'w') as f:
            json.dump(ratios, f)
    rid = ratios.index(ratio_str)

    labels[idx] = [x, y, w, rid]
    if idx not in labelled_idxs:
        labelled_idxs.append(idx)

    # Append to CSV
    # Ensure header exists if file was just created
    if os.path.getsize(labels_file) == 0:
        with open(labels_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'x', 'y', 'width', 'ratio'])
    with open(labels_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([idx, x, y, w, rid])

    return jsonify(
        total_labels=len(labels),
        ratio_stats=_get_ratio_stats(labels)
    )


@app.route('/navigate', methods=['POST'])
def navigate():
    global current_image_index
    data = request.json
    action = data.get('action')

    if action == 'next_labelled':
        if not labelled_idxs:
            return jsonify(error='No labelled images'), 400
        nxt = [i for i in labelled_idxs if i > current_image_index]
        current_image_index = nxt[0] if nxt else labelled_idxs[0]
    elif action == 'prev_labelled':
        if not labelled_idxs:
            return jsonify(error='No labelled images'), 400
        prev = [i for i in labelled_idxs if i < current_image_index]
        current_image_index = prev[-1] if prev else labelled_idxs[-1]
    else:
        return jsonify(error='Invalid action'), 400

    return jsonify(
        index=current_image_index,
        total_labels=len(labels),
        ratio_stats=_get_ratio_stats(labels)
    )


def main(project: str, stage: int = None):
    global image_dir, image_files, labels, labels_file
    global ratios, ratios_file, current_image_index, labelled_idxs

    if stage is None:
        stage = find_latest_stage(project)

    stage_dir = f"stage_{stage}"
    image_dir = os.path.join(project, stage_dir)
    if not os.path.exists(image_dir):
        print(f"Error: Stage directory '{stage_dir}' not found in {project}")
        return

    # Load images
    image_files = [f for f in os.listdir(image_dir)
                   if f.lower().endswith(('.png','.jpg','.jpeg','.gif','.bmp'))]
    current_image_index = 0 if image_files else -1

    # Setup labels file
    labels_file = os.path.join(project, f'stage_{stage}_crop_labels.csv')
    # Create CSV with header if not exists
    if not os.path.exists(labels_file):
        with open(labels_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'x', 'y', 'width', 'ratio'])
    labels.clear()
    if os.path.exists(labels_file):
        with open(labels_file, 'r', newline='') as f:
            reader = csv.reader(f)
            # Skip header
            next(reader, None)
            for row in reader:
                img_idx, xs, ys, ws, rid = row
                idx_i = int(img_idx)
                labels[idx_i] = [float(xs), float(ys), float(ws), int(rid)]

    labelled_idxs = list(labels.keys())

    # Load or init ratios
    ratios_file = os.path.join(project, f'stage_{stage}_crop_ratios.json')
    if os.path.exists(ratios_file):
        with open(ratios_file, 'r') as f:
            ratios = json.load(f)
    else:
        ratios = []

    print(f"Starting app: {len(image_files)} images, {len(labels)} labels, {len(ratios)} ratios.")
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Launch web interface for labeling images.")
    parser.add_argument('--project', type=str, required=True,
                        help='Project root containing image stage directories')
    parser.add_argument('--stage', type=int, default=None,
                        help='Stage number (e.g., 1). If omitted, uses latest.')
    args = parser.parse_args()
    main(args.project, args.stage)
