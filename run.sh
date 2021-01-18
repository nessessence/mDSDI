# for i in {1..1}; do
#     python main.py --config "algorithms/DSDI/configs/DomainNet_sketch.json" --exp_idx $i --gpu_idx "0"
# done

# tensorboard --logdir=/mnt/vinai/DSDI/algorithms/DSDI/results/tensorboards/MNIST_2

rm -r algorithms/DSDI/results/checkpoints/*
rm -r algorithms/DSDI/results/logs/*
rm -r algorithms/DSDI/results/plots/*
rm -r algorithms/DSDI/results/tensorboards/*