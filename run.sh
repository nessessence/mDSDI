for i in {1..1}; do
    python main.py --config "algorithms/DSDI_AE/configs/MNIST.json" --exp_idx $i
done

# tensorboard --logdir=/mnt/vinai/DSDI_deploy/algorithms/DSDI/results/tensorboards/MNIST_1

# rm -r algorithms/DSDI_AE/results/checkpoints/*
# rm -r algorithms/DSDI_AE/results/logs/*
# rm -r algorithms/DSDI_AE/results/plots/MNIST_1/*
# rm -r algorithms/DSDI_AE/results/plots/MNIST_2/*
# rm -r algorithms/DSDI_AE/results/plots/MNIST_3/*
# rm -r algorithms/DSDI_AE/results/plots/MNIST_4/*
# rm -r algorithms/DSDI_AE/results/plots/MNIST_5/*
# rm -r algorithms/DSDI_AE/results/tensorboards/*