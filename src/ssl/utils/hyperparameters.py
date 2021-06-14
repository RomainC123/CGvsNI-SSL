METHODS_DEFAULT = {
    'TemporalEnsembling': {
        'alpha': 0.6,
        'max_unsup_weight': 30.
    },
    'MeanTeacher': {
        'ema_teacher': 0.999,
        'max_unsup_weight': 30.
    },
    'OnlySup': {}
}

OPTIMIZERS_DEFAULT = {
    'Adam': {  # Learning rate default is in the argparse
        'beta1': 0.9,
        'beta2': 0.999
    }
}

# ------------------------------------------------------------------------------

RAMP_UP_MULT = 5
RAMP_DOWN_MULT = 12.5

SCHEDULES_DEFAULT = {
    'lr': {
        'ramp_up_epochs': 80,
        'ramp_up_mult': RAMP_UP_MULT,
        'ramp_down_epochs': 50,
        'ramp_down_mult': RAMP_DOWN_MULT,
        'lower': 0.,
        'upper': 1.
    },
    'beta1': {
        'ramp_up_epochs': 0,
        'ramp_up_mult': RAMP_UP_MULT,
        'ramp_down_epochs': 50,
        'ramp_down_mult': RAMP_DOWN_MULT,
        'lower': 0.55,
        'upper': 1.
    },
    'unsup_weight': {
        'ramp_up_epochs': 80,
        'ramp_up_mult': RAMP_UP_MULT,
        'ramp_down_epochs': 0,
        'ramp_down_mult': RAMP_DOWN_MULT,
        'lower': 0.,
        'upper': 1.
    }
}
