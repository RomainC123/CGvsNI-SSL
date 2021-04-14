from .tools import WeightSchedule

LR_SCHEDULE = WeightSchedule(ramp_up_epochs=80, ramp_up_mult=5, ramp_down_epochs=50, ramp_down_mult=12.5)
B1_SCHEDULE = WeightSchedule(ramp_up_epochs=0, ramp_up_mult=0, ramp_down_epochs=0, ramp_down_mult=0)
UNSUP_WEIGHT_SCHEDULE = WeightSchedule(ramp_up_epochs=80, ramp_up_mult=5, ramp_down_epochs=0, ramp_down_mult=0)
