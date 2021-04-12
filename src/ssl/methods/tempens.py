class TemporalEnsembling(SSLMethod):

    def __init__(self, **kwargs):

        super(TemporalEnsembling, self).__init__(hyperparameters)

    def _set_hyperparameters(self, hyperparameters):
        self.alpha = hyperparameters['alpha']
        self.ramp_epochs = hyperparameters['unsup_loss_ramp_up_epochs']
        self.ramp_mult = hyperparameters['unsup_loss_ramp_up_mult']
        self.unsup_loss_max_weight = hyperparameters['unsup_loss_max_weight']
        self.unsup_weight_schedule =

    def _get_hyperparameters_info(self):
        infos = 'Method: Temporal Ensembling\n'
        infos += f'Alpha: {self.alpha}\n'
        infos += 'Unsupervised loss max weight (uncorrected): {:.1f}\n'.format(self.unsup_loss_max_weight)
        infos += f'Ramp epochs: {self.ramp_epochs}\n'
        infos += f'Ramp mult: {self.ramp_mult}\n'

        return infos

    def _init_vars(self, epochs):
        self.y_ema = torch.zeros(self.nb_img_train, self.nb_classes).float()
        self.unsup_weight = torch.autograd.Variable(torch.FloatTensor([0.]), requires_grad=False)
