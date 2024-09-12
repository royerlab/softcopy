import os
import json

def process_zarr_directory(zarr_dir):
    """Process a .zarr directory to extract 'daxi' from .zattrs and save to daxi.json."""
    zattrs_path = os.path.join(zarr_dir, '.zattrs')
    if not os.path.isfile(zattrs_path):
        print(f"No .zattrs file found in {zarr_dir}")
        return
    
    # Read .zattrs
    with open(zattrs_path, 'r') as file:
        try:
            zattrs = json.load(file)
        except json.JSONDecodeError:
            print(f"Error decoding JSON in {zattrs_path}")
            return
    
    # Extract 'daxi' key
    daxi_data = zattrs.get('daxi')
    if daxi_data is None:
        print(f"'daxi' key not found in {zattrs_path}")
        return
    
    # Write 'daxi' data to daxi.json
    daxi_json_path = os.path.join(zarr_dir, 'daxi.json')
    with open(daxi_json_path, 'w') as file:
        json.dump(daxi_data, file, indent=4)
    print(f"Created {daxi_json_path} with 'daxi' data.")

def find_and_process_zarr_folders(root_folder):
    """Recursively find and process all .zarr directories."""
    for root, dirs, _ in os.walk(root_folder):
        for dir_name in dirs:
            if dir_name.endswith('.zarr'):
                zarr_dir = os.path.join(root, dir_name)
                process_zarr_directory(zarr_dir)

# Adjust the root folder path as needed
root_folder_path = '/hpc/projects/group.royer/imaging/misc/preliminary/2024_09_05_daxi_2_martinblanco_drosophila'
find_and_process_zarr_folders(root_folder_path)