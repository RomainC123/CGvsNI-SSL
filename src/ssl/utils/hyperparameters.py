METHODS_DEFAULT = {
    'TemporalEnsembling': {
        'alpha': 0.6,
        'unsup_loss_ramp_up_epochs': 10,
        'unsup_loss_ramp_up_mult': 5,
        'unsup_loss_max_weight': 30.
    }
}

OPTIMIZERS_DEFAULT = {
    'Adam': {
        'max_lr': 0.001,
        'beta1': 0.9,
        'beta2': 0.999
    }
}

# ------------------------------------------------------------------------------

METHODS_SEARCH = {
    'TemporalEnsembling': {
        'alpha': [0.6],
        'unsup_loss_ramp_up_epochs': [5],
        'unsup_loss_ramp_up_mult': [2, 5],
        'unsup_loss_max_weight': [20.]
    }
}
