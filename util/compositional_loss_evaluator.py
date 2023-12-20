from sentence_transformers.evaluation import SentenceEvaluator
import numpy as np
import logging
import os
import csv
from typing import List
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F

from tqdm import tqdm

from info_nce import InfoNCE, info_nce

logger = logging.getLogger(__name__)


class CompositionalLossEvaluator(SentenceEvaluator):
    """
    Computes the mean squared error (x100) between the computed sentence embedding
    and some target sentence embedding.

    The MSE is computed between ||teacher.encode(source_sentences) - student.encode(target_sentences)||.

    For multilingual knowledge distillation (https://arxiv.org/abs/2004.09813), source_sentences are in English
    and target_sentences are in a different language like German, Chinese, Spanish...

    :param source_sentences: Source sentences are embedded with the teacher model
    :param target_sentences: Target sentences are ambedding with the student model.
    :param show_progress_bar: Show progress bar when computing embeddings
    :param batch_size: Batch size to compute sentence embeddings
    :param name: Name of the evaluator
    :param write_csv: Write results to CSV file
    """
    def __init__(self, target_sentences: DataLoader, teacher_model = None, show_progress_bar: bool = False, batch_size: int = 32, name: str = '', write_csv: bool = True):
        #self.source_embeddings = teacher_model.encode(source_sentences, show_progress_bar=show_progress_bar, batch_size=batch_size, convert_to_numpy=True)

        self.target_sentences = target_sentences
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name

        self.loss = InfoNCE(negative_mode='paired')

        self.csv_file = "mse_diff_evaluation_" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps", "MSE"]
        self.write_csv = write_csv

    def __call__(self, model, output_path, epoch  = -1, steps = -1):
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        sum = 0
        count = 0
        for feats, label in tqdm(self.target_sentences):
            reps = [model.encode(feat, convert_to_tensor=True) for feat in feats]
            #loss = self.loss_function(features, label).detach().cpu().numpy()
            batch_size = reps[0].shape[0]
            rep_hypo = reps[0]  # batch * 256
            pos_reps = reps[1:(batch_size+1)] # batch
            neg_reps = reps[(batch_size+1):]  # batch

            # sum each path and stack paths
            rep_pos = torch.stack([torch.sum(rep_pos, dim=0) for rep_pos in pos_reps], dim=0)   # batch * 256
            rep_neg = torch.stack([torch.sum(rep_neg, dim=0) for rep_neg in neg_reps], dim=0)   # batch * 256

            # compute loss
            loss = self.loss(rep_hypo, rep_pos, rep_neg.unsqueeze(1)).cpu().numpy()
            """
            pos_dist = F.mse_loss(rep_hypo, rep_pos, reduction='none').sum(dim=1)
            neg_dist = F.mse_loss(rep_hypo, rep_neg, reduction='none').sum(dim=1)

            loss = F.relu(pos_dist - neg_dist).mean().cpu().numpy()
            """

            sum += loss
            count += 1

        """
        for triplet in tqdm(self.target_sentences):
            #{'query': 'sent', 'positive': ['sents', 'x'], 'negative': ['y', 'z']}
            rep_hypo = model.encode(triplet['query'], convert_to_numpy=False)
            rep_pos = model.encode(triplet['positive'], convert_to_tensor=True)
            rep_neg = model.encode(triplet['negative'], convert_to_tensor=True)

            rep_pos = torch.sum(rep_pos, dim=0)   # 256
            rep_neg = torch.sum(rep_neg, dim=0)

            pos_dist = F.pairwise_distance(rep_hypo, rep_pos, p=2).square().cpu().numpy()
            neg_dist = F.pairwise_distance(rep_hypo, rep_neg, p=2).square().cpu().numpy()

            loss = np.maximum(0, pos_dist - neg_dist)

            sum += loss
            count += 1
        """

        mse = sum / count
        mse *= 100

        #mse = ((self.source_embeddings - target_embeddings)**2).mean()

        logger.info("MSE evaluation (lower = better) on "+self.name+" dataset"+out_txt)
        logger.info("MSE (*100):\t{:4f}".format(mse))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, mse])

        return -mse #Return negative score as SentenceTransformers maximizes the performance


