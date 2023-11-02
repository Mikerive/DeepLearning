import os
import shutil

def undo_sorting(src_dir, dest_dir):
    classes = [d for d in os.listdir(dest_dir) if os.path.isdir(os.path.join(dest_dir, d))]
    
    for class_name in classes:
        for subset in ['training', 'validation', 'testing']:
            subset_dir = os.path.join(src_dir, subset, class_name)
            files = [f for f in os.listdir(subset_dir) if os.path.isfile(os.path.join(subset_dir, f))]
            
            for file in files:
                dest_path = os.path.join(dest_dir, class_name, file)
                src_path = os.path.join(subset_dir, file)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.move(src_path, dest_path)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(script_dir, 'data_sorted')
    dest_dir = os.path.join(script_dir, 'data_input')
    
    undo_sorting(src_dir, dest_dir)

if __name__ == '__main__':
    main()
