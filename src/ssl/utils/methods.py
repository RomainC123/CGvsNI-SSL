from ..methods.tempens import TemporalEnsembling
from ..methods.meanteach import MeanTeacher
from ..methods.only_sup import OnlySup
from ..methods.full_sup import FullSup
from ..methods.vat import VAT

METHODS = {
    'TemporalEnsembling': TemporalEnsembling,
    'MeanTeacher': MeanTeacher,
    'OnlySup': OnlySup,
    'FullSup': FullSup,
    'VAT': VAT
}
