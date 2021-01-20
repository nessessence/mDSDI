for i in {1..1}; do
    python main.py --config "algorithms/DSDI/configs/DomainNet_infograph.json" --exp_idx $i --gpu_idx "0"
done

# tensorboard --logdir=/mnt/vinai/DSDI/algorithms/DSDI_debug/results/tensorboards/Colored_MNIST_1

# rm -r algorithms/DSDI/results/checkpoints/*
# rm -r algorithms/DSDI/results/logs/*
# rm -r algorithms/DSDI/results/plots/*
# rm -r algorithms/DSDI/results/tensorboards/*