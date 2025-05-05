## Evaluation Directions

Use the scripts from `diffusionsfm/eval` to evaluate the performance of the dense model on the CO3D dataset:

```
python -m diffusionsfm.eval.eval_jobs --eval_path output/multi_diffusionsfm_dense --use_submitit
```

**Note:** The `use_submitit` flag is optional. If you have a SLURM system available, enabling it will dispatch jobs in parallel across available GPUs, significantly accelerating the evaluation process.

The expected output at the end of evaluating the dense model is:

```
N=                2     3     4     5     6     7     8
Seen R        0.926 0.941 0.946 0.950 0.953 0.955 0.955
Seen CC       1.000 0.956 0.934 0.924 0.917 0.911 0.907
Seen CD       0.023 0.023 0.026 0.026 0.028 0.031 0.030
Seen CD_Obj   0.040 0.037 0.033 0.032 0.032 0.032 0.033
Unseen R      0.913 0.928 0.938 0.945 0.950 0.951 0.953
Unseen CC     1.000 0.926 0.884 0.870 0.864 0.851 0.847
Unseen CD     0.024 0.024 0.025 0.024 0.025 0.026 0.027
Unseen CD_Obj 0.028 0.023 0.022 0.022 0.023 0.021 0.020
```

This reports rotation and camera center accuracy, as well as Chamfer Distance on both all points (CD) and foreground points (CD_Obj), evaluated on held-out sequences from both seen and unseen object categories using varying numbers of input images. Performance is averaged over five runs to reduce variance.

Note that minor variations in the reported numbers may occur due to randomness in the evaluation and inference processes.