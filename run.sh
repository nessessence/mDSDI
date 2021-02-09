for i in {1..1}; do
     python main.py --config "algorithms/AGG/configs/Rotated_F_MNIST_AUG.json" --exp_idx $i --gpu_idx "0"
done

# tensorboard --logdir=/mnt/vinai/DSDI/algorithms/DSDI/results/tensorboards/PACS_photo_1

# rm -r algorithms/DSDI_debug/results/checkpoints/*
# rm -r algorithms/DSDI_debug/results/logs/*
# rm -r algorithms/DSDI_debug/results/plots/*
# rm -r algorithms/DSDI_debug/results/tensorboards/*