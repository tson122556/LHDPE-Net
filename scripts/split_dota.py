from ultralytics.data.split_dota import split_test, split_trainval
import numpy as np



# split train and val set, with labels.
split_trainval(
    data_root="/home/sdb/pk/datasets/DOTAv1/",
    save_dir="/home/sdb/pk/datasets/DOTAv1_split_1024/",
    rates=[1.0,],  # multiscale
    gap=200,
)
# split test set, without labels.
split_test(
    data_root="/home/sdb/pk/datasets/DOTAv1",
    save_dir="/home/sdb/pk/datasets/DOTAv1_split_1024",
    rates=[1.0,],  # multiscale
    gap=200,
)
