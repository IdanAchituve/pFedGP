import os
import torch.nn as nn
import matplotlib
if "DISPLAY" not in os.environ or "localhost" in os.environ["DISPLAY"]:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from utils import *
import logging
import torch.nn.functional as F


class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, probs, labels, path=None, color='darkblue', temp=1.0, apply_log=True):

        # To use temperature:
        if apply_log:
            probs = F.softmax(torch.log(probs) / temp, dim=-1)  # probs to logits
        else:
            probs = F.softmax(probs / temp, dim=-1)  # probs to logits

        confidences, predictions = torch.max(probs, 1)
        accuracies = predictions.eq(labels)
        max_err = torch.tensor([0.0])

        targets = labels.long()
        y_one_hot = to_one_hot(targets)
        bri = (torch.norm(probs - y_one_hot, dim=1) ** 2).mean()

        ece = torch.zeros(1, device=probs.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

                if torch.abs(avg_confidence_in_bin - accuracy_in_bin) > max_err:
                    max_err = torch.abs(avg_confidence_in_bin - accuracy_in_bin)

        if path is not None:
            plot_calibration_error(probs, targets, path, color)
        return ece, max_err, bri


def plot_calibration_error(probs, targets, path, color='darkblue'):

    confidences = probs.max(-1).values.detach().numpy()
    accuracies = probs.argmax(-1).eq(targets).numpy()

    n_bins = 15
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    max_err = 0.0

    targets = targets.long()
    y_one_hot = to_one_hot(targets)
    bri = (torch.norm(probs - y_one_hot, dim=1) ** 2).mean()

    plot_acc = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) * (confidences <= bin_upper)
        prop_in_bin = in_bin.astype(np.float32).mean()

        if prop_in_bin > 0.0:
            accuracy_in_bin = accuracies[in_bin].astype(np.float32).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            if np.abs(avg_confidence_in_bin - accuracy_in_bin) > max_err:
                max_err = np.abs(avg_confidence_in_bin - accuracy_in_bin)

            plot_acc.append(accuracy_in_bin)
        else:
            plot_acc.append(0.0)

    plt.figure(figsize=(4, 4))
    plt.rcParams.update({'font.size': 18})

    plt.xticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], size=15)
    plt.yticks(ticks=[0.2, 0.4, 0.6, 0.8, 1.0], size=15)

    props = dict(boxstyle='round', facecolor='white', alpha=1.)

    plt.bar(
        bin_lowers, plot_acc, bin_uppers[0], align="edge", linewidth=1, edgecolor='k', color=color
    )
    plt.plot([0.0, 1.0], [0.0, 1.0], c="orange", lw=2)
    plt.text(
        0.05,
        0.73,
        "ECE:  {:0.3f}\nMCE: {:0.3f}\nBRI:  {:0.3f}".format(
            ece, max_err, detach_to_numpy(bri).astype(np.float32).item()
        ),
        fontsize=16,
        bbox=props
    )

    plt.xlim((0, 1.))
    plt.ylim((0, 1.))
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")

    logging.info(path)
    logging.info(bin_uppers)
    logging.info(bin_lowers)
    logging.info(plot_acc)

    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()