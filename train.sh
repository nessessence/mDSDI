# for i in {1..5}; do
#      taskset -c "51" python main.py --config "algorithms/mDSDI/configs/PACS_photo.json" --exp_idx $i --gpu_idx "1"
# done

# tensorboard --logdir "/home/ubuntu/mDSDI/algorithms/mDSDI/results/tensorboards/PACS_photo_1"
# python utils/tSNE_plot.py --plotdir "algorithms/mDSDI/results/plots/PACS_photo_1/"

# rm -r algorithms/mDSDI/results/checkpoints/*
# rm -r algorithms/mDSDI/results/logs/*
# rm -r algorithms/mDSDI/results/plots/*
# rm -r algorithms/mDSDI/results/tensorboards/*