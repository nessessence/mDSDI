for i in {1..1}; do
    python main.py --config "algorithms/DSDI/configs/PACS_sketch.json" --exp_idx $i
done

# tensorboard --logdir=/home/ubuntu/DSDI/algorithms/DSDI/results/tensorboards/PACS_art_2