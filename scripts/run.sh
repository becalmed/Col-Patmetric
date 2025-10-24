python compare_with_disturb.py \
  --ref /home/fangjingwu/data/dataset/test_dataset/upper/paired_image \
  --gen /home/fangjingwu/data/dataset/test_dataset/upper/ours_test_res \
  --out /home/fangjingwu/data/dataset/test_dataset/upper/ours_metric_res_with_perturb_metrics \
  --perturb_dir /home/fangjingwu/data/dataset/test_dataset/upper/perturbed_refs \
  --patch_size 32 --region upper --perturb rotate --angle 2 \
  --Emax 20 --b1 0.7 --b2 0.3 \
  --compute_folder_metrics