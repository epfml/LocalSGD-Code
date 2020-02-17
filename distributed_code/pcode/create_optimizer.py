# -*- coding: utf-8 -*-
from pcode.optim.sgd import SGD

from pcode.optim.local_sgd import LocalSGD
from pcode.optim.local_ef_sign_sgd import Local_EFSignSGD
from pcode.optim.local_sign_sgd import Local_SignSGD
from pcode.optim.sign_sgd import SignSGD
from pcode.optim.ef_sign_sgd import EF_SignSGD
from pcode.optim.dgc import DGC


def define_optimizer(conf, model):
    # define the param to optimize.
    params = [
        {
            "params": [value],
            "name": key,
            "weight_decay": conf.weight_decay if "bn" not in key else 0.0,
            "param_size": value.size(),
            "nelement": value.nelement(),
        }
        for key, value in model.named_parameters()
    ]

    # define the optimizer.
    if conf.optimizer == "sgd":
        optim_class = SGD
    elif conf.optimizer == "dgc":
        optim_class = DGC
    elif conf.optimizer == "local_sgd":
        optim_class = LocalSGD
    elif conf.optimizer == "sign_sgd":
        optim_class = SignSGD
    elif conf.optimizer == "ef_sign_sgd":
        optim_class = EF_SignSGD
    elif conf.optimizer == "local_sign_sgd":
        optim_class = Local_SignSGD
    elif conf.optimizer == "local_ef_sign_sgd":
        optim_class = Local_EFSignSGD
    else:
        raise NotImplementedError

    optimizer = optim_class(
        params,
        lr=conf.lr,
        momentum=conf.momentum_factor,
        nesterov=conf.use_nesterov,
        conf=conf,
    )
    return optimizer
