python scripts/generate_heatmaps.py \
    /h/pwilson/projects/medAI/projects/prostnfound/logs/test/prostnfound_plus_final/optimum/checkpoint.pth \
    /h/pwilson/projects/medAI/data/scratch/hmaps_final \
    --patient_ids UA-157 UA-032 UA-028 UA-006 UA-014 UA-069 \
    --core_ids UA-157-010 UA-032-014 UA-028-008 UA-006-005 UA-014-009 UA-069-012 \
    --style miccai \
    --apply_prostate_mask \
    --mode one_image_per_core 