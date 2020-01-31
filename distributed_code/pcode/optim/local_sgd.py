# -*- coding: utf-8 -*-
from copy import deepcopy

import torch
from torch.optim.optimizer import Optimizer, required

import pcode.optim.utils as utils
import pcode.utils.communication as comm
from pcode.utils.sparsification import get_n_bits
from pcode.utils.tensor_buffer import TensorBuffer


class LocalSGD(Optimizer):
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
        super(LocalSGD, self).__init__(params, defaults)

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

    def _init_consensus(self):
        params, _ = comm.get_data(
            self.param_groups, self.param_names, is_get_grad=False
        )
        self.consensus_params_tb = deepcopy(TensorBuffer(params))

    def __setstate__(self, state):
        super(LocalSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None, **kargs):
        with kargs["timer"]("sync.local_update", epoch=self.conf.epoch_):
            utils.apply_gradient(
                self.param_groups, self.state, apply_grad_to_model=True
            )

        with kargs["timer"]("sync.sync_and_update", epoch=self.conf.epoch_):
            # enter the global sync if it satisfies the condition.
            if (
                self.conf.epoch_ < self.turn_on_local_step_from_epoch
                or self.conf.local_index % self.local_step == 0
            ):
                # get parmas.
                params, _ = comm.get_data(
                    self.param_groups, self.param_names, is_get_grad=False
                )
                params_tb = TensorBuffer(params)

                # get params_diff.
                param_diff = self.consensus_params_tb.buffer - params_tb.buffer
                # sync the directions.
                param_diff = self.world_aggregator._agg(
                    param_diff, "avg", distributed=self.conf.distributed
                )

                # unpack the synced info and update the consensus params.
                self.consensus_params_tb.buffer.add_(-1.0, param_diff)

                # consistent the local models by assigning the consensus params.
                self.consensus_params_tb.unpack(params)

                # Get n_bits to transmit.
                n_bits = get_n_bits(param_diff)
            else:
                n_bits = 0
        return n_bits

