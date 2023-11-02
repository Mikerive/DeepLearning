import os
import numpy as np
import matplotlib.pyplot as plt

def count_images(dir_path):
    return len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])

def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_sorted')
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    subsets = ['training', 'validation', 'test']
    
    for subset in subsets:
        class_sizes = [count_images(os.path.join(data_dir, class_name, subset)) for class_name in classes]
        total_images = sum(class_sizes)
        # Plotting
        plt.figure(figsize=(10, 6))
        bars = plt.bar(classes, class_sizes, color=plt.cm.viridis(np.linspace(0, 1, len(classes))))
        plt.xlabel('Classes')
        plt.ylabel('Number of Images')
        plt.title(f'Distribution of {total_images} Images per Class ({subset.capitalize()})')
        
        # Creating legend labels
        legend_labels = [f'{classes[i]}: {class_sizes[i]}' for i in range(len(classes))]
        plt.legend(bars, legend_labels, loc='upper right')
        
        plt.show()

if __name__ == '__main__':
    main()
