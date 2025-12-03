python compare_with_disturb.py \
  --ref /home/fangjingwu/data/related_works/vitonhd/test/image \
  --gen /home/fangjingwu/data/dataset/vitonhdres/ours \
  --out /home/fangjingwu/data/dataset/cache_test \
  --perturb_dir /home/fangjingwu/data/dataset/cache_test/perturb \
  --patch_size 32 --region upper --perturb elastic \
  --Emax 20 --b1 0.7 --b2 0.3 \
  --compute_folder_metrics