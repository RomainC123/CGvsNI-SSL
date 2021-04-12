from ssl.models.VGG import VGGContainer
from ssl.optimizers.adam import AdamOptimizer
import os
from ssl.utils.paths import TRAINED_MODELS_PATH

model = VGGContainer('normal')
optimizer = AdamOptimizer(max_lr=0.001, beta1=0.9, beta2=0.999)

print(optimizer(model, 40, 300))
