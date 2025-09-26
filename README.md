# GNN

## Introduction

This repo contains codes for generating datasets and training GNN models.

## Instruction

### Step 1. Clone this repo

```shell
git clone git@github.com:hdj020402/GNN.git
```

### Step 2. Create conda environment

```shell
conda create -n <newenv> python=3.12
```

- If you are going to run the GNN code on a device with GPU, you can run the following command.

    ```shell
    cd gnn
    pip install -r envs/requirements-gpu.txt
    ```

- If you don't have a GPU, run the following command.

    ```shell
    cd gnn
    pip install -r envs/requirements-cpu.txt
    ```

- **Notice:** the creation of environment may fail due to network instability, especially when installing pytorch or pytorch-related packages. If the creation is disrupted, install the packages seperately.

### Step 3. Set parameters

1. Create a new file called `model_parameters.yml` in the same folder as `main.py`.
2. `model_parameters_example.yml` gives an example of the parameter file. Copy **all the terms** in the example file to `model_parameters.yml` created previously.
3. Modify the parameters to meet your demand. Each parameter is explained as follows:
    - General
        - `jobtype`: the type of your job, can be any words
        - `mode`: corresponding to different functions; choose among `training`, `hparam_tuning`, `feature_filtration`, `prediction` and `fine-tuning`
            - `training`: train a model with given parameters
            - `hparam_tuning`: train multiple models with different combinations of hyper-parameters in order to choose the best one according to models' performance; to use this function, file `hparam_tuning.yml` is a must, and the content of this file will be explained in detail in [point 4](#hparam_tuning)
            - `feature_filtration`: train multiple models with different combinations of features in order to choose the best one according to models' performance
            - `prediction`: predict the target in a new dataset with a pre-trained model
            - `fine-tuning`: perform training with a pre-trained model on a different dataset
        - `feature_filter_mode`: method of screening the best combination of features; choose among `one_by_one`, `file` and `null`
            - `one_by_one`: `node_attr_list`, `edge_attr_list` and `graph_attr_list` will be merged into a new list, and each time, one feature will be removed from the new list to form a new feature combination
            - `file`: not implemented; you can record multiple combination of features you want to test in `feature_filter.yml`, and train the models one by one; to use this function, file `feature_filter.yml` is a must, and the content of this file will be explained in detail in [point 5](#feature_filtration)
            - `null`: no screening task
        - `seed`: integer, to generate reproducible random splitted datasets and prediction results
        - `GPU_memo_frac`: decimal, the proportion of GPU memory (or CUDA memory) that can be allocated and used by a single process
    - Dataset (relative paths are recommended)
        - `path`: the path of a directory to store processed data

            ```plain text
            <path>/
            └── processed/
                ├── graph_data.pt
                ├── model_parameters.yml
                ├── pre_filter.pt
                └── pre_transform.pt
            ```

        - `sdf_file`: the path of a `.sdf` file, which sequentially records the coordinates and bonding information of structures
        - `node_attr_file`: the path of a `.json` file or a `.pkl` file, which sequentially records the node information of structures; can be `null`
            - e.g.

                ```json
                {
                    "charge": [[-0.2, 0.05, -0.19], [0.33, -0.81, 0.24]],
                    "dispersion": [[-1.77, -2.16, -2.09], [-1.76, -3.11, -2.98]]
                }
                ```

            - Each value corresponds to an atomic node, which means they should be recorded in the same order as the atoms  in the `.sdf` file.
        - `edge_attr_file`: the path of a `.json` file or a `.pkl` file, which sequentially records the edge information of structures (similar to `node_attr_file`); can be `null`
            - e.g.
                ```json
                {
                    "order": [[1, 1, 2, 2, 1, 1], [3, 3, 1, 1, 1, 1, 2, 2]],
                    "energy": [[-2.3, -2.3, 2.2, 2.2, 1.0, 1.0], [-3.56, -3.56, -1.42, -1.42, -2.67, -2.67]]
                }
                ```
            - Each value corresponds to an edge, which means they should be recorded in the same order as the bonds appear in the result of `mol.GetBonds()`; to generate an undirected graph, each edge must be listed twice in succession but in opposite directions, so the edge attributes should also be included twice
        - `graph_attr_file`: the path of a `.csv` file, which sequentially records the molecular features and the targets of structures; molecular features are not necessary, while targets are must-have contents
            - e.g.

                |feature 1|feature 2|...|feature n|target|
                |-|-|-|-|-|
                |0.1|0.2|...|0.4|1|
                |0.2|0.4|...|0.8|2|
                |...|...|...|...|...|
                |1|2|...|4|10|

        - `atom_type`: list, to identify all unique element types present in the structures
        - `default_node_attr`: dict[str, bool], you can choose whether to include each default node attribute
            - `ele_type`: one hot, determine the type of element
            - `atomic_number`: the atomic number of an atom
            - `aromatic`: whether the atom is aromatic or not
            - `num_neighbors`: number of connected atoms
            - `num_hs`: number of connected Hs
        - `default_edge_attr`: dict, you can choose whether to include each default edge attribute
            - `edge_type`: one hot, determine the bond order
            - `bond_length`: dict
                - `power`: list, the format of the elements in the list must be `r^n`, where n is a float; e.g. `[r^1, r^-2, r^-0.33]`
                - `threshold`: float, if the distance between two nodes is greater than the threshold, the bond length will be infinite; can be `null`, which means there is no threshold
        - `node_attr_list`: list, to select the node features you need; can be `null` or `[]`
        - `edge_attr_list`: list, to select the edge features you need; can be `null` or `[]`
        - `graph_attr_list`: list, to select the molecular features you need; can be `null` or `[]`
        - `node_attr_filter`: not implemented
        - `edge_attr_filter`: not implemented
        - `pos`: not implemented
        - `target_type`: choose among `graph`, `node` and `edge`
        - `target_list`: list, to determine the target
        - `target_transform`: apply mathematical transformation to target; usually used when the distribution of data is uneven; choose among `LN`, `LG`, `E^-x` and `null`
            - `LN`: $y' = \ln{y}$
            - `LG`: $y' = \log{y}$
            - `E^-x`: $y' = e^{-y}$
        - `batch_size`: integer
        - `num_workers`: integer
        - `split_method`: choose between `random` and `manual`
        - `split_file`: the path of a `.npy` file, which uses `np.array` to record the index of data in the order of training dataset, validation dataset and test dataset; if you choose `manual` in `split_method`, you will have to offer a valid path here
        - `train_size`: decimal, the proportion of training dataset; if you choose `random` in `split_method`, you will have to offer a valid proportion here
        - `val_size`: decimal, the proportion of validation dataset; if you choose `random` in `split_method`, you will have to offer a valid proportion here
    - Model
        - `pretrained_model`: the path of a `.pth` file; in `training` mode, if the path is valid, then training will continue on the basis of this model; in `prediction` or `fine-tuning` mode, pre-trained model is a must
        - `dim_linear`: integer, the dimension of nn hidden layers
        - `dim_conv`: integer, the dimension of conv
        - `processing_steps`: integer, number of processing steps in Set2Set
        - `mp_times`: integer, number of times for message passing
        - `loss_fn`: choose among `MAE` and `MSE`
        - `optimizer`: choose among `Adam`, `AdamW` and `SGD`; the string must be exactly the same as the attribute of `torch.optim`
        - `lr`: learning rate
        - `schduler`:
            - `type`: `ReduceLROnPlateau`, `null`, etc.
            - `factor`: decimal
            - `patience`: integer
            - `min_lr`: decimal
    - Training
        - `accumulation_step`: integer, the number of steps between gradient accumulation
        - `epoch_num`: integer
        - `output_step`: integer, the number of steps between printing output information
        - `model_save_step`: integer, the number of steps between generating checkpoints
        - `early_stopping`: decide whether to stop training earlier than determined `epoch_num` according to the value of `val_loss`
            - `patience`: integer
            - `delta`: float
        - `criteria_list`: list; criteria of estimating the model; you can choose one or more criteria mentioned below, but `[]` or `null` is not allowed
            - `MAE`: Mean Absolute Error
            $$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
            - `MSE`: Mean Squared Error
            $$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
            - `RMSD`: Root Mean Squared Deviation
            $$RMSD = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
            - `R2`: Coefficient of Determination
            $$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$
            - `AARD`: Average Absolute Relative Deviation
            $$AARD(\%) = \frac{100}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$
    - Prediction
        - `dataset_range`: choose among `train`, `val`, `test` and `whole`
            - `train`, `val` and `test`: in case you want to predict a specific part of the whole dataset, you can select one of the three options; the sub-datasets will be generated according to the `split_method` as mentioned before
            - `whole`: use the whole dataset for prediction
<a id="hparam_tuning"></a>

4. `hparam_tuning_example.yml` gives an example of the hparam_tuning file. You can alse modify the parameters to meet your demand. Each parameter is explained as follows:
    - General
        - `optuna`: hparam tuning is realized using `optuna`
            - `sampler`: algorithm used for hparam tuning; users can look up the source code for the most suitable one
                - `type`: name of the algorithm; the string must be exactly the same as the attribute of `optuna.samplers`
                - `kwargs`: users can modify whichever parameter in the function by providing `keyword: argument`
            - `pruner`: algorithm used for terminating trials with bad performance; users can look up the source code for the most suitable one
                - `type`: name of the algorithm; the string must be exactly the same as the attribute of `optuna.pruners`
                - `kwargs`: users can modify whichever parameter in the function by providing `keyword: argument`
            - `direction`: choose between `minimize` and `maximize`
            - `n_trials`: integer
            - `continue_trials`:
                `continue`: boolean, to decide whether to continue a hparam-tuning task
                `storage`: the path of a `.db` file; the path must be in the format of `sqlite:///{$regular_path}`
                `study_name`: `hptuning_{$jobtype}`; if `null`, the first study in the `storage` will be chosen
    - hyperparameters
        - `hparam`: must be one of the keys in `model_parameters.yml`
            - `type`: choose among `int`, `float`, `uniform`, `discrete_uniform`, `loguniform` and `categorical`
            - `kwargs`: users can modify whichever parameter in the function by providing `keyword: argument`
<a id="feature_filtration"></a>

5. `feature_filtration_example.yml` gives an example of the feature_filtration file.

### Step 4. Train your model

Make sure you are in the `gnn/` folder and run the following command.

```shell
nohup python main.py > Training_Recording/recording.log 2>&1 &
```

With this command, you can train the model in the background. The folder `Training_Recording/` will be generated automatically and you can check the `.log` file for error messages.

### Step 5. Post-processing

The results will be output to `Training_Recording/`. The structure of `Training_Recording/` is as follows.

```plain text
gnn/
├── Training_Recording/
|   ├── <jobtype>/
|   |   ├── <TIME>/
|   |   |   ├── Model/
|   |   |   |   ├── checkpoint/
|   |   |   |   |   ├── ckpt_<TIME>_050.pth
|   |   |   |   |   ├── ckpt_<TIME>_100.pth
|   |   |   |   |   └── ...
|   |   |   |   └── best_model_<loss_fn>_<TIME>.pth
|   |   |   ├── Plot/
|   |   |   |   ├── best_model_<loss_fn>_<TIME>_test.png
|   |   |   |   ├── best_model_<loss_fn>_<TIME>_train_test.png
|   |   |   |   ├── best_model_<loss_fn>_<TIME>_train.png
|   |   |   |   └── ...
|   |   |   ├── gpu_monitor.log
|   |   |   ├── model_parameters.yml
|   |   |   └── training_<TIME>.log
|   |   └── recording/
|   └── recording.log
└── ...
```

Results of the same type of job will be classified to a folder called `<jobtype>/`. Then, different training tasks will have a unique folder named after a specific time in order to distinguish among one another.

Checkpoints are stored in `Model/checkpoint/` according to the `model_save_step` set in `model_parameters.yml`, while two best models based on AARD and MAE respectively are stored in `Model/` directly.

The scatter plots of best models predicting the data of different datasets and the `<data>-epoch` plots are stored in `Plot/`, which can offer you an intuitive understanding of the results.

File `training_<TIME>.log` records the parameters and the basic information of datasets. Besides, it records the training information of specific epochs according to the `output_step` set in `model_parameters.yml`. This information is used to generate the `<data>-epoch` plots. At the end of the file, the information of the best epoch based on two criteria are extracted from previous content.
