import pandas as pd

from config import DATASETS, METRICS_DIR
from src.data_loader import load_dataset
from src.features import (
    extract_raw_pixels,
    extract_hog_features,
    extract_lbp_features,
)
from src.train_eval import train_svm, evaluate_model
from src.visualization import plot_confusion_matrix


def run_single_experiment(dataset_name, feature_name, x_train, y_train, x_test, y_test, class_names):
    print(f"\n--- {dataset_name} | {feature_name} başladı ---")

    if feature_name == "raw_pixel":
        X_train = extract_raw_pixels(x_train)
        X_test = extract_raw_pixels(x_test)

    elif feature_name == "hog":
        X_train = extract_hog_features(x_train)
        X_test = extract_hog_features(x_test)

    elif feature_name == "lbp":
        X_train = extract_lbp_features(x_train)
        X_test = extract_lbp_features(x_test)

    else:
        raise ValueError(f"Unknown feature name: {feature_name}")

    print("Train feature shape:", X_train.shape)
    print("Test feature shape:", X_test.shape)

    model = train_svm(X_train, y_train)

    accuracy, precision, recall, macro_f1, cm = evaluate_model(
        model,
        X_test,
        y_test,
    )

    plot_confusion_matrix(
        cm=cm,
        dataset_name=dataset_name,
        feature_name=feature_name,
        class_names=class_names,
    )

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Macro F1:", macro_f1)

    return {
        "dataset": dataset_name,
        "feature_method": feature_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "macro_f1": macro_f1,
        "train_feature_shape": str(X_train.shape),
        "test_feature_shape": str(X_test.shape),
    }


def run_all_experiments():
    results = []
    feature_methods = ["raw_pixel", "hog", "lbp"]

    for dataset_name in DATASETS:
        print(f"\n==============================")
        print(f"DATASET: {dataset_name}")
        print(f"==============================")

        x_train, y_train, x_test, y_test, class_names = load_dataset(dataset_name)

        for feature_name in feature_methods:
            result = run_single_experiment(
                dataset_name,
                feature_name,
                x_train,
                y_train,
                x_test,
                y_test,
                class_names,
            )
            results.append(result)

    results_df = pd.DataFrame(results)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = METRICS_DIR / "summary_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\nTüm deneyler tamamlandı.")
    print(f"Sonuç dosyası kaydedildi: {output_path}")
    print("\n", results_df)

    return results_df