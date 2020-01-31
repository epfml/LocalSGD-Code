# -*- coding: utf-8 -*-
from copy import deepcopy

import torch
from torch.optim.optimizer import Optimizer, required

import pcode.optim.utils as utils
import pcode.utils.communication as comm
from pcode.utils.sparsification import get_n_bits
from pcode.utils.tensor_buffer import TensorBuffer


class Local_EFSignSGD(Optimizer):
    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        conf=None,
        model=None,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(Local_EFSignSGD, self).__init__(params, defaults)

        # store the whole training arguments.
        self.conf = conf
        self.rank = conf.graph.rank
        self.neighbors_info = conf.graph.get_neighborhood()
        self.local_step = conf.local_step
        self.turn_on_local_step_from_epoch = conf.turn_on_local_step_from

        # define the aggregator.
        self.world_aggregator = comm.get_aggregators(
            cur_rank=self.rank,
            world=conf.graph.ranks,
            neighbors_info=dict(
                (rank, 1.0 / conf.graph.n_nodes) for rank in conf.graph.ranks
            ),
            aggregator_type="centralized",
        )

        # define sorted param names.
        self.param_names = list(
            enumerate([group["name"] for group in self.param_groups])
        )

        # initialize the concensus
        self._init_consensus()
        self._init_memory()

    def _init_consensus(self):
        params, _ = comm.get_data(
            self.param_groups, self.param_names, is_get_grad=False
        )
        self.consensus_params_tb = deepcopy(TensorBuffer(params))

    def _init_memory(self):
        params, self.shapes = comm.get_data(
            self.param_groups, self.param_names, is_get_grad=False
        )
        self.memory_tb = TensorBuffer(params)
        self.memory_tb.buffer = torch.zeros_like(self.memory_tb.buffer)

    def __setstate__(self, state):
        super(Local_EFSignSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None, **kargs):
        # do the local update steps.
        with kargs["timer"]("sync.local_update", epoch=self.conf.epoch_):
            utils.apply_gradient(
                self.param_groups, self.state, apply_grad_to_model=True
            )

        # enter the global sync if it satisfies the condition.
        if (
            self.conf.epoch_ < self.turn_on_local_step_from_epoch
            or self.conf.local_index % self.local_step == 0
        ):
            with kargs["timer"]("sync.get_params", epoch=self.conf.epoch_):
                # get parmas.
                params, _ = comm.get_data(
                    self.param_groups, self.param_names, is_get_grad=False
                )
                params_tb = TensorBuffer(params)

            # get the params difference w.r.t. previous synced model.
            local_scale, local_sign = [], []
            for consensus_param, param, memory in zip(
                self.consensus_params_tb, params_tb, self.memory_tb
            ):
                # add memory to the model difference.
                memory.data.copy_(consensus_param - param + memory)
                # compress.
                _local_scale, _local_sign = scaled_sign(memory)
                # update memory.
                memory.data.copy_(memory - _local_scale * _local_sign)
                # store local scales and local sign.
                local_scale.append(_local_scale)
                local_sign.append(_local_sign)

            # concat the update magnitude and directions.
            magnitudes_tb = TensorBuffer(local_scale)
            directions_tb = TensorBuffer(local_sign)

            # sync and decompress.
            with kargs["timer"]("sync.sync_and_decompress", epoch=self.conf.epoch_):
                # sync the directions.
                directions_tb.buffer = self.world_aggregator._agg(
                    directions_tb.buffer, "avg", distributed=self.conf.distributed
                )
                magnitudes_tb.buffer = self.world_aggregator._agg(
                    magnitudes_tb.buffer, "avg", distributed=self.conf.distributed
                )

            # unpack the synced info and update the consensus params.
            with kargs["timer"]("sync.update_consensus", epoch=self.conf.epoch_):
                for update_magnitude, update_direction, consensus_param in zip(
                    magnitudes_tb, directions_tb, self.consensus_params_tb
                ):
                    consensus_param.add_(-1.0, update_direction.mul(update_magnitude))

            # consistent the local models by assigning the consensus params.
            self.consensus_params_tb.unpack(params)
            n_bits = get_n_bits(directions_tb.buffer) + get_n_bits(magnitudes_tb.buffer)
        else:
            n_bits = 0
        return n_bits


def scaled_sign(x, name=None):
    """
    :param x: torch Tensor
    :return: The sign tensor scaled by it's L1 norm divided by the number of elements
    """
    _scale = x.norm(p=1) / x.numel()
    _sign = torch.sign(x)

    return _scale, _sign
