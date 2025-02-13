import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import norm
from configs import model_to_run, save_dir
def parse_tcav_file(filename):
    concept_scores = {}
    current_concept = None
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
       
            if line.startswith("Concept ="):
                parts = line.split("=")
                if len(parts) >= 2:
                    current_concept = parts[1].strip()
                    if current_concept not in concept_scores:
                        concept_scores[current_concept] = []
            
            elif "TCAV Score =" in line:
             
                m = re.search(r"TCAV Score = ([0-9.]+)", line)
                if m and current_concept is not None:
                    score = float(m.group(1))
                    concept_scores[current_concept].append(score)
    return concept_scores

def compute_statistics(concept_scores):

    stats = {}
    for concept, scores in concept_scores.items():
        arr = np.array(scores)
        mean = np.mean(arr)
        variance = np.var(arr)
        stats[concept] = {"mean": mean, "variance": variance}
    return stats

def save_statistics(stats, output_filename):

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("Concept\tMean\tVariance\n")
        for concept, stat in stats.items():
            f.write(f"{concept}\t{stat['mean']:.3f}\t{stat['variance']:.3f}\n")

def plot_violin(concept_scores, output_filename):

    data = []
    for concept, scores in concept_scores.items():
        for s in scores:
            data.append((concept, s))
    df = pd.DataFrame(data, columns=["Concept", "TCAV_Score"])
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Concept", y="TCAV_Score", data=df)
    plt.title("Violin Plot of TCAV Scores per Concept")
    plt.ylabel("TCAV Score")
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()

def plot_bell_curves(concept_scores, output_filename):

    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10.colors 
    for idx, (concept, scores) in enumerate(concept_scores.items()):
        arr = np.array(scores)
        mean = np.mean(arr)
        std = np.std(arr)

        if std == 0:
            std = 0.01
        x_vals = np.linspace(mean - 3*std, mean + 3*std, 100)
        y_vals = norm.pdf(x_vals, mean, std)
        plt.plot(x_vals, y_vals, 
                 label=f"{concept} (mean={mean:.2f}, std={std:.2f})",
                 color=colors[idx % len(colors)])
    plt.title("Bell Curves (Normal Distributions) of TCAV Scores per Concept")
    plt.xlabel("TCAV Score")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()

def process_tcav_file(save_path, input_filename, stats_output_filename, violin_output_filename, bell_output_filename):
    input_filename = os.path.join(save_path, input_filename)
    stats_output_filename = os.path.join(save_path, stats_output_filename)
    violin_output_filename = os.path.join(save_path, violin_output_filename)
    bell_output_filename = os.path.join(save_path, bell_output_filename)

    concept_scores = parse_tcav_file(input_filename)

    stats = compute_statistics(concept_scores)
    
 
    save_statistics(stats, stats_output_filename)
    
 
    plot_violin(concept_scores, violin_output_filename)
    
  
    plot_bell_curves(concept_scores, bell_output_filename)
    
    print("处理完成，统计结果和图片均已保存。")


if __name__ == "__main__":

    # save_path = os.path.join(save_dir, model_to_run, "recostructed_results")
    save_path = os.path.join(save_dir, model_to_run, "original_results")
    input_filename = "log.txt"             
    stats_output_filename = "tcav_stats.txt"   
    violin_output_filename = "tcav_violin.png" 
    bell_output_filename = "tcav_bell.png"    
    
    process_tcav_file(save_path, input_filename, stats_output_filename, violin_output_filename, bell_output_filename)
