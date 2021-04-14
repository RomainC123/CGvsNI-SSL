HYPERPARAMETERS_DEFAULT = {
    'TemporalEnsembling': {
        'alpha': 0.6,
        'unsup_loss_ramp_up_epochs': 10,
        'unsup_loss_ramp_up_mult': 5,
        'unsup_loss_max_weight': 30.
    }
}

OPTIMIZER_PARAMS = {
    'Adam': {
        'max_lr': 0.003,
        'ramp_up_epochs': 0,
        'ramp_up_mult': 5,
        'ramp_down_epochs': 0,
        'ramp_down_mult': 12.5
    }
}


HYPERPARAMETERS_SEARCH = {
    'TemporalEnsembling': {
        'alpha': [0.6],
        'unsup_loss_ramp_up_epochs': [5],
        'unsup_loss_ramp_up_mult': [2, 5],
        'unsup_loss_max_weight': [20.]
    }
}
