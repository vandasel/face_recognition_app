import matplotlib.pyplot as plt
import re
import seaborn as sns

def dataset_plotter(data):
    sns.set_theme(style="whitegrid")
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
        x_bottom = list(sorted_r.keys())[-11:-1]
        y_bottom = list(sorted_r.values())[-11:-1]

        plt.figure(figsize=(16, 6))
        plt.barh(x_top, y_top, color='skyblue')

        plt.title(f'Top 10 People in {names[i]} Dataset', fontsize=16)
        plt.xlabel('Count', fontsize=12)
        

        plt.xlim(min(y_top) - 2, max(y_top) + 1)
        plt.xticks(fontsize=10)

        plt.savefig(f'plots/{names[i]}_top10.png')
        plt.close()

        plt.figure(figsize=(16, 6))
        plt.barh(x_bottom, y_bottom, color='lightcoral')

        plt.title(f'Bottom 10 People in {names[i]} Dataset', fontsize=16)
        plt.xlabel('Count', fontsize=12)
        
        plt.xlim(min(y_bottom) - 2, max(y_bottom) + 1)
        plt.xticks(fontsize=10)

        plt.savefig(f'plots/{names[i]}_bottom10.png')
        plt.close()
