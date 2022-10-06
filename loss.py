import torch
import torch.nn as nn
import hparams as hp

class MyHingeLoss(nn.Module):
    def __init__(self):
        super(MyHingeLoss, self).__init__()

    def forward(self, output, target):
        # Rescale target from {0, 1} to {-1, 1}
        target = target * 2 - 1
        return torch.mean(torch.maximum(1 - output*target, torch.zeros(output.shape, device=output.device)))


class DNNLoss(nn.Module):
    def __init__(self):
        super(DNNLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.hinge_loss = MyHingeLoss()

    def forward(self, mel, mel_postnet, duration_predicted, f0_predicted, voiced_predicted, mel_target, duration_predictor_target, f0_target, voiced_target):
        if mel.shape[1] > mel_target.shape[1]:
            mel = mel[:,0:mel_target.shape[1]]
            mel_postnet = mel_postnet[:, 0:mel_target.shape[1]]
            f0_predicted = f0_predicted[:,0:mel_target.shape[1]]
            voiced_predicted = voiced_predicted[:, 0:mel_target.shape[1]]
        if mel.shape[1] < mel_target.shape[1]:
            mel_target = mel_target[:,0:mel.shape[1]]
            f0_target = f0_target[:,0:mel.shape[1]]
            voiced_target = voiced_target[:,0:mel.shape[1]]

        mel_target.requires_grad = False
        mel_loss = self.mse_loss(mel, mel_target)
        mel_postnet_loss = self.mse_loss(mel_postnet, mel_target)

        duration_predictor_target.requires_grad = False
        duration_predicted.masked_fill(duration_predictor_target == 0, 0.0)
        duration_predictor_loss = self.mse_loss(duration_predicted * hp.duration_loss_norm,
                                                duration_predictor_target.float() * hp.duration_loss_norm)

        f0_target.requires_grad = False
        if hp.guide_f0_with_voiced_targets:
            f0_predicted.masked_fill(~voiced_target, 0.0)
            f0_target.masked_fill(~voiced_target, 0.0)
        f0_loss = self.mse_loss(f0_predicted, f0_target)

        voiced_target.requires_grad = False
        voiced_loss = self.bce_loss(voiced_predicted, voiced_target.float())
        #voiced_loss = self.hinge_loss(voiced_predicted, voiced_target.float())

        return mel_loss, mel_postnet_loss, duration_predictor_loss, f0_loss, voiced_loss
