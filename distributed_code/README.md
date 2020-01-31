# Getting started
Our experiments heavily rely on `Docker` and `Kubernetes`. For the detailed experimental environment setup, please refer to dockerfile under the `environments` folder.


## Use case of distributed training (centralized)
Some simple explanation of the arguments used in the code.
* Arguments related to *distributed training*:
    * The `n_mpi_process` and `n_sub_process` indicates the number of nodes and the number of GPUs for each node. The data-parallel wrapper is adapted and applied locally for each node.
        * Note that the exact mini-batch size for each MPI process is specified by `batch_size`, while the mini-batch size used for each GPU is `batch_size/n_sub_process`.
    * The `world` describes the GPU topology of the distributed training, in terms of all GPUs used for the distributed training.
    * The `hostfile` from `mpi` specifies the physical location of the MPI processes.
    * We provide two use cases here:
        * `n_mpi_process=2`, `n_sub_process=1` and `world=0,0` indicates that two MPI processes are running on 2 GPUs with the same GPU id. It could be either 1 GPU at the same node or two GPUs at different nodes, where the exact configuration is determined by `hostfile`.
        * `n_mpi_process=2`, `n_sub_process=2` and `world=0,1,0,1` indicates that two MPI processes are running on 4 GPUs and each MPI process uses GPU id 0 and id 1 (on 2 nodes).
* Arguments related to *communication compression*:
    * The `graph_topology` 
    * The `optimizer` will decide the type of distributed training, e.g., centralized SGD, decentralized SGD
    * The `comm_op` specifies the communication compressor we can use, e.g., `sign+norm`, `random-k`, `top-k`.
* Arguments related to *learning*:
    * The `lr_schedule_scheme` and `lr_change_epochs` indicates that it is a stepwise learning rate schedule, with decay factor `10` for epoch `150` and `225`.
    * The `lr_scaleup`, `lr_warmup` and `lr_warmup_epochs` will decide if we want to scale up the learning rate, or warm up the learning rate.

### Examples
The script below trains `ResNet-20` with `CIFAR-10`, as an example of centralized training algorithm `(post-)local SGD`.
]