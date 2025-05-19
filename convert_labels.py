import os
import csv
import argparse
from train import find_album_directories

def convert_label_format(directory: str, dry_run: bool = False) -> int:
    """
    Convert crop_labels.csv from the old format (x1, y1, x2, y2) to the new format (x1, y1, height, ratio).
    
    Args:
        directory: Root directory containing album directories or a specific album directory
        dry_run: If True, only print actions without making changes
        
    Returns:
        Number of files converted
    """
    # Check if this is a single album directory with a crop_labels.csv file
    if os.path.exists(os.path.join(directory, 'crop_labels.csv')):
        album_dirs = [directory]
    else:
        # Find all album directories with label data
        album_dirs = find_album_directories(directory)
    
    print(f"Found {len(album_dirs)} album directories with label data")
    
    if len(album_dirs) == 0:
        print("No valid album directories found. Exiting.")
        return 0
    
    converted_count = 0
    
    for album_dir in album_dirs:
        album_name = os.path.basename(album_dir)
        labels_file = os.path.join(album_dir, 'crop_labels.csv')
        
        if not os.path.exists(labels_file):
            print(f"No labels file found in {album_name}. Skipping.")
            continue
        
        try:
            # Read the CSV file to determine its format
            with open(labels_file, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader, None)  # Get header row
                
                # Check if it's already in the new format
                if header and len(header) == 5:
                    if header[3] == 'height' and header[4] == 'ratio':
                        print(f"Labels in {album_name} are already in the new format. Skipping.")
                        continue
                    
                    # It's in the old format (x1, y1, x2, y2)
                    print(f"Converting labels in {album_name} from old format to new format...")
                    
                    # Read all rows
                    rows = list(reader)
                    
                    if dry_run:
                        print(f"  Would convert {len(rows)} labels in {album_name} (DRY RUN)")
                        converted_count += 1
                        continue
                    
                    # Convert and write back
                    with open(labels_file, 'w', newline='') as f_out:
                        writer = csv.writer(f_out)
                        writer.writerow(['img_name', 'x1', 'y1', 'height', 'ratio'])  # New header
                        
                        for row in rows:
                            if len(row) == 5:  # Ensure row has 5 columns (img_name, x1, y1, x2, y2)
                                img_name = row[0]
                                x1 = float(row[1])
                                y1 = float(row[2])
                                x2 = float(row[3])
                                y2 = float(row[4])
                                
                                # Calculate height
                                height = y2 - y1
                                
                                # Calculate aspect ratio and determine ratio code
                                width = x2 - x1
                                aspect_ratio = width / height if height > 0 else 1.0
                                
                                # Assign ratio code
                                if abs(aspect_ratio - 1.0) < 0.1:
                                    ratio_code = 0  # 1:1
                                elif abs(aspect_ratio - (2/3)) < 0.1:
                                    ratio_code = 1  # 2:3
                                elif abs(aspect_ratio - (3/2)) < 0.1:
                                    ratio_code = 2  # 3:2
                                else:
                                    # Default to the closest ratio
                                    ratios = [1.0, 2/3, 3/2]
                                    differences = [abs(aspect_ratio - r) for r in ratios]
                                    ratio_code = differences.index(min(differences))
                                
                                # Write the converted row
                                writer.writerow([img_name, x1, y1, height, ratio_code])
                            
                        print(f"  Converted {len(rows)} labels in {album_name}")
                        converted_count += 1
                else:
                    print(f"Unexpected CSV format in {album_name}. Skipping.")
        
        except Exception as e:
            print(f"Error converting labels in {album_name}: {str(e)}")
    
    return converted_count


def main():
    parser = argparse.ArgumentParser(description="Convert crop labels from old format (x1, y1, x2, y2) to new format (x1, y1, height, ratio)")
    parser.add_argument("directory", type=str, help="Root directory containing album directories or a specific album directory")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without making changes")
    
    args = parser.parse_args()
    
    print(f"Starting label format conversion in {args.directory}")
    print(f"Dry run: {args.dry_run}")
    
    converted = convert_label_format(args.directory, args.dry_run)
    
    if args.dry_run:
        print(f"Would convert {converted} label files (DRY RUN)")
    else:
        print(f"Converted {converted} label files")
    
    print("Conversion completed.")


if __name__ == "__main__":
    main()
