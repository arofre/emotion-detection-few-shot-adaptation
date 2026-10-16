import os
import pandas as pd
from numpy import mean
reanchored = pd.read_csv("results/cosine_happiness/resnet50_reanchored_base/classification_report.csv", index_col=0)

reports = {}
for folder in [20, 50, 100, 200, 500]:
    k_report = pd.read_csv(f"results/cosine_happiness/resnet50_k{folder}_stage2/classification_report.csv", index_col=0)
    f1_diff = k_report.loc["macro avg", "f1-score"] - reanchored.loc["macro avg", "f1-score"]
    recall_diff = {label: k_report.loc[label, "recall"] - reanchored.loc[label, "recall"] 
                   for label in k_report.index if label not in ["macro avg", "accuracy", "weighted avg", "happiness"]}

    reports[folder] = (f1_diff, recall_diff)

    print(f"Results for k={folder}:")
    print("Average recall difference:")
    print(mean(list(recall_diff.values())))
    print("Excluded class recall")
    print(f"happiness: {k_report.loc['happiness', 'recall']:.4f}")
    
    #print("Recall differences per class:")
    #for label, diff in recall_diff.items():
    #    print(f"  {label}: {diff:.4f}")