import os
import pandas as pd
from numpy import mean

import pandas as pd
from numpy import mean
emotion = "happiness"

reanchored = pd.read_csv(f"results/cosine_{emotion}/resnet50_reanchored_base/classification_report.csv", index_col=0)

base = pd.read_csv(
    f"results/replay_{emotion}/resnet50_stage1/classification_report.csv",
    index_col=0
)

print("Results for exemplar replay linear classifier")
reports = {}
for folder in [20, 50, 100, 200, 500]:
    k_report = pd.read_csv(f"results/cosine_{emotion}/resnet50_k{folder}_stage2/classification_report.csv", index_col=0)
    f1_diff = k_report.loc["macro avg", "f1-score"] - reanchored.loc["macro avg", "f1-score"]
    recall_diff = {label: k_report.loc[label, "recall"] - reanchored.loc[label, "recall"] 
                   for label in k_report.index if label not in ["macro avg", "accuracy", "weighted avg", emotion]}

    reports[folder] = (f1_diff, recall_diff)

    print(f"Results for k={folder}:")
    print(f"Average recall difference: {mean(list(recall_diff.values())):.4f}")
    print(f"{emotion} recall: {k_report.loc[emotion, 'recall']:.4f}")
    print()
    #print("Recall differences per class:")
    #for label, diff in recall_diff.items():
    #    print(f"  {label}: {diff:.4f}")

import pandas as pd
from numpy import mean

base = pd.read_csv(
    f"results/replay_{emotion}/resnet50_stage1/classification_report.csv",
    index_col=0
)

reports = {}
print("Results for exemplar replay linear classifier")
for folder in [20, 50, 100, 200, 500]:
    for r in [0.5, 1, 2, 5, 10]:
        k_report = pd.read_csv(
            f"results/replay_{emotion}/resnet50_k{folder}_r{r}_stage2/classification_report.csv",
            index_col=0
        )

        f1_diff = (
            k_report.loc["macro avg", "f1-score"]
            - base.loc["macro avg", "f1-score"]
        )

        excluded = ["macro avg", "accuracy", "weighted avg", emotion]

        recall_diff = {
            label: k_report.loc[label, "recall"] - base.loc[label, "recall"]
            for label in k_report.index
            if label not in excluded
        }

        avg_recall_diff = mean(list(recall_diff.values()))

        reports[(folder, r)] = (f1_diff, recall_diff)

        print(f"Results for k={folder}, r={r}:")
        print(f"Average recall difference: {avg_recall_diff:.4f}")
        print(f"{emotion} recall: {k_report.loc[emotion, 'recall']:.4f}")
        print()