from utils import *
import matplotlib.pyplot as plt

total_epochs = 200

schedule = WeightSchedule(ramp_up_epochs=80, ramp_up_mult=5, ramp_down_epochs=50, ramp_down_mult=12.5)

list_weights = []

for i in range(total_epochs):
    list_weights.append(schedule.step(total_epochs=total_epochs))

plt.plot(list(range(total_epochs)), list_weights)
plt.show()
