from ..methods.tempens import TemporalEnsembling
from ..methods.meanteach import MeanTeacher
from ..methods.only_sup import OnlySup

METHODS = {
    'TemporalEnsembling': TemporalEnsembling,
    'MeanTeacher': MeanTeacher,
    'OnlySup': OnlySup
}
