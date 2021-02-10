# meta-Domain Specific-Domain Invariant (mDSDI)

### Guideline
To prepare: (download, unzip the datasets and pretrained models)
- bash setup.sh

To train model: (select different settings by editing in /configs/..json and train.sh, results are stored in /results/logs/)
- bash train.sh

To visualize objective functions: ()
- tensorboard --logdir=/mnt/vinai/mDSDI/algorithms/DSDI/results/tensorboards/PACS_photo_1

To plot t-SNE: ()
- python utils/tSNE_plot.py
