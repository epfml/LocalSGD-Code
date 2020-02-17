# Don't Use Large Mini-batches, Use Local SGD
We present here the code of the experimental parts of the paper [Don't Use Large Mini-batches, Use Local SGD](https://openreview.net/forum?id=B1eyO1BFPr).

Abstract:
Mini-batch stochastic gradient methods (SGD) are state of the art for distributed training of deep neural networks. 
Drastic increases in the mini-batch sizes have lead to key efficiency and scalability gains in recent years. 
However, progress faces a major roadblock, as models trained with large batches often do not generalize well, i.e. they do not show good accuracy on new data.
As a remedy, we propose a post-local SGD and show that it significantly improves the generalization performance compared to large-batch training on standard benchmarks while enjoying the same efficiency (time-to-accuracy) and scalability. We further provide an extensive study of the communication efficiency vs. performance trade-offs associated with a host of local SGD variants. 


# Code usage
We rely on `Docker` for our experimental environments. Please refer to the folder `distributed_code/environments/docker` for more details.

The script below trains `ResNet-20` with `CIFAR-10`, as an example of centralized training algorithm `(post) local SGD`.
For the detailed instructions and more examples, please refer to the file `distributed_code/README.md`.
```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $HOME/conda/envs/pytorch-py3.6/bin/python run.py \
    --arch resnet20 --optimizer local_sgd \
    --avg_model True --experiment demo --manual_seed 6 \
    --data cifar10 --pin_memory True \
    --batch_size 128 --base_batch_size 64 --num_workers 2 \
    --num_epochs 300 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --n_mpi_process 16 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 \
    --on_cuda True --use_ipc False \
    --lr 0.1 --lr_scaleup True --lr_warmup True --lr_warmup_epochs 5 \
    --lr_scheduler MultiStepLR --lr_decay 0.1 --lr_milestones 150,225 \
    --local_step 16 --turn_on_local_step_from 150 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --hostfile hostfile --graph_topology complete --track_time True --display_tracked_time True \
    --python_path $HOME/conda/envs/pytorch-py3.6/bin/python --mpi_path $HOME/.openmpi/
```

# Reference
If you use this code, please cite the following [paper](https://openreview.net/forum?id=B1eyO1BFPr)

```
@inproceedings{lin2020dont,
    title={Don't Use Large Mini-batches, Use Local {SGD}},
    author={Tao Lin and Sebastian U. Stich and Kumar Kshitij Patel and Martin Jaggi},
    booktitle={ICLR - International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=B1eyO1BFPr}
}
```
