import os
import shutil
import random

def create_directory_structure(base_dir, classes):
    for subset in ['training', 'validation', 'testing']:
        for class_name in classes:
            os.makedirs(os.path.join(base_dir, subset, class_name), exist_ok=True)

def split_and_move_files(src_dir, dest_dir, class_name):
    files = [f for f in os.listdir(os.path.join(src_dir, class_name)) if os.path.isfile(os.path.join(src_dir, class_name, f))]
    random.seed(42)  # to make the split reproducible
    random.shuffle(files)
    
    # Change the split to use fixed numbers instead of percentages
    train_files = files[:1000]
    val_files = files[1000:1400]
    test_files = files[1400:1800]
    
    for file in train_files:
        shutil.move(os.path.join(src_dir, class_name, file), os.path.join(dest_dir, 'training', class_name, file))
    for file in val_files:
        shutil.move(os.path.join(src_dir, class_name, file), os.path.join(dest_dir, 'validation', class_name, file))
    for file in test_files:
        shutil.move(os.path.join(src_dir, class_name, file), os.path.join(dest_dir, 'testing', class_name, file))

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(script_dir, 'data_input')
    dest_dir = os.path.join(script_dir, 'data_sorted')
    classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    
    create_directory_structure(dest_dir, classes)
    
    for class_name in classes:
        split_and_move_files(src_dir, dest_dir, class_name)

if __name__ == '__main__':
    main()
