# -*- coding: utf-8 -*-
import os
import copy

import torch
import torch.distributed as dist

import parameters as para
import pcode.distributed_running_cv as flow
import pcode.create_dataset as create_dataset
import pcode.create_model as create_model
from pcode.create_dataset import load_data_batch
import pcode.create_metrics as create_metrics
from pcode.utils.stat_tracker import RuntimeTracker
import pcode.utils.topology as topology
import pcode.utils.op_files as op_files


def evaluate(
    model, criterion, metrics, data_loader, label="train_loader", their_conf=dict()
):
    print(f"evaluate for {label}.")
    # define stat.
    tracker_te = RuntimeTracker(metrics_to_track=metrics.metric_names)

    # switch to evaluation mode
    model.eval()

    for _input, _target in data_loader[label]:
        # load data and check performance.
        _input, _target = load_data_batch(conf, _input, _target)

        with torch.no_grad():
            flow.inference(model, criterion, metrics, _input, _target, tracker_te)
    current_eval = tracker_te()
    current_eval["conf"] = their_conf
    return current_eval


def reload_model(model, ckp_path):
    model_clone = copy.deepcopy(model)
    checkpoint_path = os.path.join(ckp_path, "0", "checkpoint.pth.tar")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print("reloaded model.")
    model_clone.load_state_dict(checkpoint["state_dict"])
    their_conf = op_files.load_pickle(os.path.join(ckp_path, "arguments.pickle"))
    return model_clone, their_conf


def iter_given_folder(model, criterion, metrics, data_loader, exp_folder_path):
    list_of_info = {}
    for _folder_path in os.listdir(exp_folder_path):
        folder_path = os.path.join(exp_folder_path, _folder_path)
        print(f"processing {folder_path}")

        try:
            # reload model.
            their_model, their_conf = reload_model(model, folder_path)

            print(
                their_conf.optimizer,
                their_conf.graph_topology,
                their_conf.n_mpi_process,
            )

            # evaluate the model and info.
            info = dict(
                (
                    label,
                    evaluate(
                        their_model, criterion, metrics, data_loader, label, their_conf
                    ),
                )
                for label in ["train_loader", "val_loader"]
            )
            list_of_info[folder_path] = info
        except RuntimeError as e:
            print(f"runtime error info={e}")
        except NotADirectoryError as e:
            print(f"runtime error info={e}")
        except FileNotFoundError as e:
            print(f"runtime error info={e}")
    print(f"get # of info = {len(list_of_info)}")
    return list_of_info


def main(conf):
    # define the graph.
    conf.distributed = False
    cur_rank = dist.get_rank() if conf.distributed else 0
    conf.graph = topology.define_graph_topology(
        graph_topology=conf.graph_topology,
        world=conf.world,
        n_mpi_process=conf.n_mpi_process,  # the # of total main processes.
        n_sub_process=conf.n_sub_process,  # the # of subprocess for each main process.
        comm_device=conf.comm_device,
        on_cuda=conf.on_cuda,
        rank=cur_rank,
    )

    # get data_loader.
    data_loader = create_dataset.define_dataset(conf, force_shuffle=True)

    # create model
    model = create_model.define_model(conf, data_loader=data_loader)

    # define the criterion and metrics.
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    criterion = criterion.cuda() if conf.graph.on_cuda else criterion
    metrics = create_metrics.Metrics(
        model.module if "DataParallel" == model.__class__.__name__ else model,
        task="classification",
    )

    list_of_info = iter_given_folder(
        model, criterion, metrics, data_loader, conf.resume
    )
    op_files.write_pickle(list_of_info, os.path.join(conf.resume, "post_eval.pickle"))


if __name__ == "__main__":
    conf = para.get_args()
    main(conf)
