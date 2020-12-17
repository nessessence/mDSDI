for i in {1..1}; do
    python main.py --config "algorithms/DSDI/configs/PACS_art.json" --exp_idx $i
done

# tensorboard --logdir=/vinai/habm1/DSDI/algorithms/DSDI/results/tensorboards/MNIST_1 
# tensorboard --logdir=/mnt/vinai/DSDI/algorithms/DSDI/results/tensorboards/PACS_art_1