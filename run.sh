# for i in {1..1}; do
#      python main.py --config "algorithms/mDSDI/configs/PACS_art.json" --exp_idx $i --gpu_idx "0"
# done

# tensorboard --logdir=/mnt/vinai/mDSDI/algorithms/DSDI/results/tensorboards/PACS_photo_1

# rm -r algorithms/mDSDI/results/checkpoints/*
# rm -r algorithms/mDSDI/results/logs/*
# rm -r algorithms/mDSDI/results/plots/*
# rm -r algorithms/mDSDI/results/tensorboards/*

# taskset -c ${p[0]}