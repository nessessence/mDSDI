# meta-Domain Specific-Domain Invariant (mDSDI)
Repository for the ICML21 submission: "Exploiting Domain-Specific Features to Enhance Domain Generalization".
![framework](gallery/framework.png)

## Guideline
### To prepare:
Install prerequisite packages:
```sh
python -m pip install -r requirements.txt
```

Download, unzip the datasets and pretrained models:
```sh
bash setup.sh
```

<img src="gallery/dataset.png" width="50%" height="50%">

### To run experiments:
Run with five different seeds:
```sh
for i in {1..5}; do
     taskset -c <cpu_index> python main.py --config <config_path> --exp_idx $i --gpu_idx <gpu_index>
done
```
where the parameters are the following:
- `<cpu_index>`: CPU index. E.g., `<cpu_index> = "1"`
- `<config_path>`: path stored configuration hyper-parameters. E.g., `<config_path> = "algorithms/mDSDI/configs/PACS_photo.json"`
- `<gpu_index>`: GPU index. E.g., `<gpu_index> = "0"`

**Note:** Select different settings by editing in `/configs/..json`, logging results are stored in `/results/logs/`

### To visualize objective functions:

```sh
tensorboard --logdir <logdir>
```
where `<logdir>`: absolute path stored TensorBoard results. E.g., `<logdir> = "/home/ubuntu/mDSDI/algorithms/mDSDI/results/tensorboards/PACS_photo_1"`

<img src="gallery/Loss.png" width="50%" height="50%">

### To plot feature representations:

```sh
python utils/tSNE_plot.py --plotdir <plotdir>
```
where `<plotdir>`: path stored results to plot. E.g., `<plotdir> = "algorithms/mDSDI/results/plots/PACS_photo_1/"`

<img src="gallery/tSNE.png" width="50%" height="50%">

**Note:** Results are stored in `/results/plots/`