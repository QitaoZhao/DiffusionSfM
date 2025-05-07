"""
python -m diffusionsfm.eval.eval_jobs --eval_path output/multi_diffusionsfm_dense --use_submitit
"""

import os
import json
import submitit
import argparse
import itertools
from glob import glob

import numpy as np
from tqdm.auto import tqdm

from diffusionsfm.dataset.co3d_v2 import TEST_CATEGORIES, TRAINING_CATEGORIES
from diffusionsfm.eval.eval_category import save_results
from diffusionsfm.utils.slurm import submitit_job_watcher


def evaluate_diffusionsfm(eval_path, use_submitit, mode):
    JOB_PARAMS = {
        "output_dir": [eval_path],
        "checkpoint": [800_000],
        "num_images": [2, 3, 4, 5, 6, 7, 8],
        "sample_num": [0, 1, 2, 3, 4],
        "category": TEST_CATEGORIES, # TRAINING_CATEGORIES + TEST_CATEGORIES,
        "calculate_additional_timesteps": [True],
    }
    if mode == "test":
        JOB_PARAMS["category"] = TEST_CATEGORIES
    elif mode == "train1":
        JOB_PARAMS["category"] = TRAINING_CATEGORIES[:len(TRAINING_CATEGORIES) // 2]
    elif mode == "train2":
        JOB_PARAMS["category"] = TRAINING_CATEGORIES[len(TRAINING_CATEGORIES) // 2:]
    keys, values = zip(*JOB_PARAMS.items())
    job_configs = [dict(zip(keys, p)) for p in itertools.product(*values)]

    if use_submitit:
        log_output = "./slurm_logs"
        executor = submitit.AutoExecutor(
            cluster=None, folder=log_output, slurm_max_num_timeout=10
        )
        # Use your own parameters
        executor.update_parameters(
            slurm_additional_parameters={
                "nodes": 1,
                "cpus-per-task": 5,
                "gpus": 1,
                "time": "6:00:00",
                "partition": "all",
                "exclude": "grogu-1-9, grogu-1-14,"
            }
        )
        jobs = []
        with executor.batch():
            # This context manager submits all jobs at once at the end.
            for params in job_configs:
                job = executor.submit(save_results, **params)
                job_param = f"{params['category']}_N{params['num_images']}_{params['sample_num']}"
                jobs.append((job_param, job))
        jobs = {f"{job_param}_{job.job_id}": job for job_param, job in jobs}
        submitit_job_watcher(jobs)
    else:
        for job_config in tqdm(job_configs):
            # This is much slower.
            save_results(**job_config)


def process_predictions(eval_path, pred_index, checkpoint=800_000, threshold_R=15, threshold_CC=0.1):
    """
    pred_index should be 1 (corresponding to T=90)
    """
    def aggregate_per_category(categories, metric_key, num_images, sample_num, threshold=None):
        """
        Aggregates one metric over all data points in a prediction file and then across categories.
        - For R_error and CC_error: use mean to threshold-based accuracy
        - For CD and CD_Object: use median to reduce the effect of outliers
        """
        per_category_values = []

        for category in tqdm(categories, desc=f"Sample {sample_num}, N={num_images}, {metric_key}"):
            per_pred_values = []

            data_path = glob(
                os.path.join(eval_path, "eval", f"{category}_{num_images}_{sample_num}_ckpt{checkpoint}*.json")
            )[0]

            with open(data_path) as f:
                eval_data = json.load(f)

            for preds in eval_data.values():
                if metric_key in ["R_error", "CC_error"]:
                    vals = np.array(preds[pred_index][metric_key])
                    per_pred_values.append(np.mean(vals < threshold))
                else: 
                    per_pred_values.append(preds[pred_index][metric_key])

            # Aggregate over all predictions within this category
            per_category_values.append(
                np.mean(per_pred_values) if metric_key in ["R_error", "CC_error"]
                else np.median(per_pred_values)  # CD or CD_Object — use median to filter outliers
            )

        if metric_key in ["R_error", "CC_error"]:
            return np.mean(per_category_values)
        else:
            return np.median(per_category_values)

    def aggregate_metric(categories, metric_key, num_images, threshold=None):
        """Aggregates one metric over 5 random samples per category and returns the final mean"""
        return np.mean([
            aggregate_per_category(categories, metric_key, num_images, sample_num, threshold=threshold)
            for sample_num in range(5)
        ])

    # Output containers
    all_seen_acc_R, all_seen_acc_CC = [], []
    all_seen_CD, all_seen_CD_Object = [], []
    all_unseen_acc_R, all_unseen_acc_CC = [], []
    all_unseen_CD, all_unseen_CD_Object = [], []

    for num_images in range(2, 9):
        # Seen categories
        all_seen_acc_R.append(
            aggregate_metric(TRAINING_CATEGORIES, "R_error", num_images, threshold=threshold_R)
        )
        all_seen_acc_CC.append(
            aggregate_metric(TRAINING_CATEGORIES, "CC_error", num_images, threshold=threshold_CC)
        )
        all_seen_CD.append(
            aggregate_metric(TRAINING_CATEGORIES, "CD", num_images)
        )
        all_seen_CD_Object.append(
            aggregate_metric(TRAINING_CATEGORIES, "CD_Object", num_images)
        )

        # Unseen categories
        all_unseen_acc_R.append(
            aggregate_metric(TEST_CATEGORIES, "R_error", num_images, threshold=threshold_R)
        )
        all_unseen_acc_CC.append(
            aggregate_metric(TEST_CATEGORIES, "CC_error", num_images, threshold=threshold_CC)
        )
        all_unseen_CD.append(
            aggregate_metric(TEST_CATEGORIES, "CD", num_images)
        )
        all_unseen_CD_Object.append(
            aggregate_metric(TEST_CATEGORIES, "CD_Object", num_images)
        )

    # Print the results in formatted rows
    print("N=           ", " ".join(f"{i: 5}" for i in range(2, 9)))
    print("Seen R       ", " ".join([f"{x:0.3f}" for x in all_seen_acc_R]))
    print("Seen CC      ", " ".join([f"{x:0.3f}" for x in all_seen_acc_CC]))
    print("Seen CD      ", " ".join([f"{x:0.3f}" for x in all_seen_CD]))
    print("Seen CD_Obj  ", " ".join([f"{x:0.3f}" for x in all_seen_CD_Object]))
    print("Unseen R     ", " ".join([f"{x:0.3f}" for x in all_unseen_acc_R]))
    print("Unseen CC    ", " ".join([f"{x:0.3f}" for x in all_unseen_acc_CC]))
    print("Unseen CD    ", " ".join([f"{x:0.3f}" for x in all_unseen_CD]))
    print("Unseen CD_Obj", " ".join([f"{x:0.3f}" for x in all_unseen_CD_Object]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_path", type=str, default=None)
    parser.add_argument("--use_submitit", action="store_true")
    parser.add_argument("--mode", type=str, default="test")
    args = parser.parse_args()

    eval_path = "output/multi_diffusionsfm_dense" if args.eval_path is None else args.eval_path
    use_submitit = args.use_submitit
    mode = args.mode

    evaluate_diffusionsfm(eval_path, use_submitit, mode)
    process_predictions(eval_path, 1)