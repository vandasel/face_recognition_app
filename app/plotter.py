import matplotlib.pyplot as plt
import re
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image
import os
from matplotlib.ticker import MaxNLocator
def dataset_plotter(data):
    sns.set_style(style="darkgrid")
    names = ["train", "val", "test"]

    for i, part in enumerate(data):
        r = {}
        for path in part:
            match = re.search(r'\/pins_(.*)\/', path)
            if match:
                group = match.group(1)
                if group not in r:
                    r[group] = 1
                else:
                    r[group] += 1

     
        sorted_r = dict(sorted(r.items(), key=lambda item: item[1]))
        x_top = list(sorted_r.keys())[0:10]
        y_top = list(sorted_r.values())[0:10]
        x_bottom = list(sorted_r.keys())[-10:]
        y_bottom = list(sorted_r.values())[-10:]

        data_combined = pd.DataFrame({
            'person': x_top + x_bottom,
            'count': y_top + y_bottom,
            'category': ['Least 10'] * 10 + ['Most 10'] * 10
        })

        plt.figure(figsize=(16, 8))
        sns.barplot(data=data_combined, y='person', x='count', hue='category', dodge=False)

        plt.title(f'Data distribution in: {names[i].title()}-Dataset', fontsize=16)
        plt.xlabel('Count', fontsize=12)
        plt.ylabel('Person', fontsize=12)

   
        plt.savefig(f'plots/{names[i]}_top_bottom.png')
        plt.close()

def plot_distances(values, correctness, plot_type,name,dist_calc):
    """
    do it
    """
    
   
    data = pd.DataFrame({
        'distance': values,
        'correctness': correctness
    })

    
    sns.set_style(style="darkgrid")

   
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='distance', hue='correctness', multiple='dodge')

    
    plt.xlabel('Distance', fontsize=12)
    plt.ylabel('Count', fontsize=12)

   
    plt.savefig(f"plots/{plot_type}_{name}_{dist_calc}.png")
    plt.close()

def save_mistakes(obj,k,filedir : str):
    img1 = Image.open(obj[1])
    img2 = Image.open(obj[0].get("documents", [[[]]])[0][0])
    img1.save(f"{filedir}/image{k}_1.png")   
    img2.save(f"{filedir}/image{k}_2.png") 
    img1.close()
    img2.close()



def histogram_plot(data):
    sns.set_style("darkgrid")
    
    group_counts = []
    
    for path in data:
        match = re.search(r'\/pins_(.*)\/', path)
        if match:
            group = match.group(1)
            group_counts.append(group)
    

    group_frequencies = [int(group_counts.count(group)) for group in set(group_counts)]

    plt.figure(figsize=(10, 6))
    sns.histplot(group_frequencies, bins=10, kde=False)
    plt.xlabel("Liczba zdjęć unikalnych osób")
    plt.ylabel("Częstotliwość") 
    plt.title("Histogram liczby zdjęć unikalnych osób")
    
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig("ludzie_histogram.png", dpi=300)
    plt.close()

paths = []
for dirpath, dirnames, filenames in os.walk("/workspaces/face_recognition_app/dataset"):
    for filename in filenames:
        paths.append(os.path.join(dirpath, filename))


histogram_plot(paths)