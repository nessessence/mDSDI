for i in {1..1}; do
    python main.py --config "algorithms/DSDI/configs/PACS_art.json" --exp_idx $i
done

# tensorboard --logdir=/home/DSDI/algorithms/DSDI/results/tensorboards/PACS_art_1