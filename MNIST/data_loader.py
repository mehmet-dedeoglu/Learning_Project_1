import torch.utils.data as data_utils
import torch
import numpy as np
import os
import glob
from PIL import Image


def get_data_loader_custom(args, size, train_classes, subsample=0):
    if type(train_classes[0]) is not list:
        train_classes = [train_classes]
    dataset_train = []
    for _a in range(len(train_classes)):
        fps_train_temp = [args.data_folder + '/' + task_ + '/Train' for task_ in os.listdir(args.data_folder)
                          if int(os.path.split(task_)[-1].split('_')[1]) in train_classes[_a]]
        dataset_hor = []
        for ell in range(len(train_classes[_a])):
            files = glob.glob(fps_train_temp[ell] + '/*')
            new = []
            for k in range(len(files)):
                data = np.asarray(Image.open(files[k]), dtype="float32")
                data = data.reshape(1, 32, 32)
                data = (data - 127.5) / 127.5  # Normalize the images to [-1, 1]
                new.append(data)
                print(str(k))
            new = np.array(new)
            dataset_hor.extend(new)
        dataset_train.append(np.array(dataset_hor))

    total_sample_num = dataset_train[0].shape[0]
    if subsample != 0:  # Can only subsample from a single item list 'dataset_train'.
        random_indices = np.asarray(np.floor(np.random.uniform(0, total_sample_num,
                                                               subsample)),  dtype='uint16')
        dataset_train[0] = dataset_train[0][random_indices]
    # Create a dataset
    train_ds_ = []
    test_ds_ = []
    for i in range(len(dataset_train)):
        train_ds_.append(torch.utils.data.DataLoader(dataset_train[i], batch_size=size, shuffle=True,
                                                     num_workers=args.workers, pin_memory=True))
    return train_ds_, test_ds_
