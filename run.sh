for i in {1..1}; do
    python main.py --config "algorithms/DSDI/configs/PACS_art.json" --exp_idx $i --gpu_idx "0"
done

# tensorboard --logdir=/mnt/vinai/DSDI/algorithms/DSDI_A/results/tensorboards/PACS_sketch_2

# rm -r algorithms/DSDI_A/results/checkpoints/*
# rm -r algorithms/DSDI_A/results/logs/*
# rm -r algorithms/DSDI_A/results/plots/*
# rm -r algorithms/DSDI_A/results/tensorboards/*