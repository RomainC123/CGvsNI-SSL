from ..methods.tempens import TemporalEnsembling
from ..methods.meanteach import MeanTeacher
from ..methods.only_sup import OnlySup
from ..method.vat import VAT

METHODS = {
    'TemporalEnsembling': TemporalEnsembling,
    'MeanTeacher': MeanTeacher,
    'OnlySup': OnlySup,
    'VAT': VAT
}
