for i in {1..1}; do
    python main.py --config "algorithms/DSDI/configs/MNIST.json" --exp_idx $i
done

# tensorboard --logdir=/home/ubuntu/DSDI/algorithms/DSDI/results/tensorboards/MNIST_1