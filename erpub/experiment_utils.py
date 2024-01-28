import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_blocking_comparison(title, thresholds, precisions, recalls, f1_scores):
    plt.plot(thresholds, precisions, label="Precision", marker="o")
    plt.plot(thresholds, recalls, label="Recall", marker="o")
    plt.plot(thresholds, f1_scores, label="F1-Score", marker="o")
    plt.legend()
    plt.title(title)
    plt.xlabel("Matching Thresholds")
    plt.ylabel("Scores")
    plt.show()


def plot_matching_accs(title, thresholds, accs):
    plt.plot(thresholds, accs, label="Accuracy", marker="o")
    plt.legend()
    plt.title(title)
    plt.xlabel("Matching Thresholds")
    plt.ylabel("Accuracy (%)")
    plt.show()


def evaluate_blocking_method(ground_truth, experiment):
    ground_truth_df = pd.read_csv(os.path.join(ground_truth, "matched_entities.csv"))
    experiment_df = pd.read_csv(os.path.join(experiment, "matched_entities.csv"))
    merged_df = pd.merge(ground_truth_df, experiment_df, how="outer", indicator=True)
    false_negatives = len(
        merged_df[merged_df["_merge"] == "left_only"].drop("_merge", axis=1)
    )
    false_positives = len(
        merged_df[merged_df["_merge"] == "right_only"].drop("_merge", axis=1)
    )
    true_positives = len(pd.merge(ground_truth_df, experiment_df))
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score


def get_accuracy_of_matches(matches_path: str, ground_truth_path: str = "labeled_entities.csv") -> float:
    ground_truth = pd.read_csv(ground_truth_path)
    matches_df = pd.read_csv(matches_path)
    acm_column, dblp_column = matches_df.keys()
    matches = 0
    for _, row in ground_truth.iterrows():
        if isinstance(row["ACM"], float): # is nan
            matches += 1 if len(matches_df[matches_df[acm_column] == row["ACM"]]) == 0 else 0
        elif isinstance(row["DBLP"], float): # is nan
            matches += 1 if len(matches_df[matches_df[dblp_column] == row["DBLP"]]) == 0 else 0
        else:
            matcher_row = matches_df[matches_df[acm_column] == row["ACM"]]
            matches += 1 if len(matcher_row) == 1 and all(matcher_row[dblp_column] == row["DBLP"]) else 0
    return matches / len(ground_truth)
