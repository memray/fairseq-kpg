# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass

import scipy.stats as stats
import numpy as np


@dataclass
class SentencePredictionConfig(FairseqDataclass):
    classification_head_name: str = field(
        default="sentence_classification_head",
        metadata={"help": "name of the classification head to use"},
    )
    regression_target: bool = field(
        default=False,
    )


@register_criterion("sentence_prediction", dataclass=SentencePredictionConfig)
class SentencePredictionCriterion(FairseqCriterion):
    def __init__(self, cfg: SentencePredictionConfig, task):
        super().__init__(task)
        self.classification_head_name = cfg.classification_head_name
        self.regression_target = cfg.regression_target

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model, "classification_heads")
            and self.classification_head_name in model.classification_heads
        ), "model must provide sentence classification head for --criterion=sentence_prediction"

        logits, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.classification_head_name,
        )
        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()

        if not self.regression_target:
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            loss = F.nll_loss(lprobs, targets, reduction="sum")

            if self.task.cfg.num_classes == 2:
                tp = ((logits[:, 0] <= logits[:, 1]) & (targets == 1)).long().sum()
                fp = ((logits[:, 0] <= logits[:, 1]) & (targets == 0)).long().sum()
                fn = ((logits[:, 0] > logits[:, 1]) & (targets == 1)).long().sum()
                tn = ((logits[:, 0] > logits[:, 1]) & (targets == 0)).long().sum()
                assert (tp + fp + tn + fn) == targets.size(0), 'invalid size'
        else:
            logits = logits.view(-1).float()
            targets = targets.float()
            loss = F.mse_loss(logits, targets, reduction="sum")

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }
        if not self.regression_target:
            preds = logits.argmax(dim=1)
            logging_output["ncorrect"] = (preds == targets).sum()

            if self.task.cfg.num_classes == 2:
                logging_output.update(tp=utils.item(tp.data) if reduce else tp.data)
                logging_output.update(fp=utils.item(fp.data) if reduce else fp.data)
                logging_output.update(fn=utils.item(fn.data) if reduce else fn.data)
                logging_output.update(tn=utils.item(tn.data) if reduce else tn.data)
        else:
            logging_output.update(x=logits.detach().cpu().numpy())
            logging_output.update(y=targets.detach().cpu().numpy())

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens
            )

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / nsentences, nsentences
            )

        metrics.log_scalar("ntokens", ntokens, 1)
        metrics.log_scalar("nsentences", nsentences, 1)
        metrics.log_scalar("sample_size", sample_size, 1)

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            tp_sum = sum(log.get('tp', 0) for log in logging_outputs)
            fp_sum = sum(log.get('fp', 0) for log in logging_outputs)
            fn_sum = sum(log.get('fn', 0) for log in logging_outputs)
            tn_sum = sum(log.get('tn', 0) for log in logging_outputs)
            if tp_sum + fp_sum + fn_sum + tn_sum > 0:
                assert tp_sum + fp_sum + fn_sum + tn_sum == sample_size, 'invalid size when aggregating'
                acc = (tp_sum + tn_sum) / sample_size
                tmp = 2 * tp_sum + fp_sum + fn_sum
                f1 = (2 * tp_sum) / tmp if tmp else 0
                tmp = (tp_sum + fp_sum) * (tp_sum + fn_sum) * (tn_sum + fp_sum) * (tn_sum + fn_sum)
                mcc = (tp_sum * tn_sum - fp_sum * fn_sum) / (tmp ** 0.5) if tmp else 0
                metrics.log_scalar("f1", f1, 1)
                metrics.log_scalar("mcc", mcc, 1)
                metrics.log_scalar("acc_f1", 0.5 * (acc + f1), 1)

        if len(logging_outputs) > 0 and 'x' in logging_outputs[0]:
            x = np.concatenate([log.get('x', np.array([])) for log in logging_outputs])
            y = np.concatenate([log.get('y', np.array([])) for log in logging_outputs])
            pearson = stats.pearsonr(x, y)[0]
            spearman = stats.spearmanr(x, y)[0]
            metrics.log_scalar("pearson", pearson, 1)
            metrics.log_scalar("spearman", spearman, 1)
            metrics.log_scalar("pearson_spearman", 0.5 * (pearson + spearman), 1)


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
