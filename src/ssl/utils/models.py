from ..models.VGG import VGGContainer
from ..models.CNN import CNNContainer
from ..models.Resnet import Resnet18Container
from ..models.SimpleNet import SimpleNetContainer
from ..models.StandardNet import StandardNetContainer
from ..models.ENet import ENetContainer

MODELS = {
    'VGG': VGGContainer,
    'CNN': CNNContainer,
    'Resnet18': Resnet18Container,
    'SimpleNet': SimpleNetContainer,
    'StandardNet': StandardNetContainer,
    'ENet': ENetContainer
}
