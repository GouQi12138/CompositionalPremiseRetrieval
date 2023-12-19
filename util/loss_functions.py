import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


from info_nce import InfoNCE, info_nce



class CompositionalLoss(nn.Module):
    """
    This class implements compositional loss. Given a pair of (hypo, path),
    the loss minimizes the distance between hypo and path.
    It compute the following loss function:

    loss = ||hypo - sum(path)||.

    :param model: SentenceTransformerModel

    Example::
        from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
        from sentence_transformers.readers import InputExample

        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        train_examples = [InputExample(texts=['Anchor 1', 'Positive 1', 'Negative 1']),
            InputExample(texts=['Anchor 2', 'Positive 2', 'Negative 2'])]
        train_dataset = SentencesDataset(train_examples, model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.TripletLoss(model=model)
    """
    def __init__(self, model: SentenceTransformer, batch=16):
        super(CompositionalLoss, self).__init__()
        self.model = model
        self.batch = batch

    def get_config_dict(self):
        return {}

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        # 1 hypo matrix + 32 path matrices
        #assert len(sentence_features) == self.batch + 1

        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]

        rep_hypo = reps[0]  # batch * 256

        # sum each path and stack paths
        rep_path = torch.stack([torch.sum(rep_path, dim=0) for rep_path in reps[1:]], dim=0)  # batch * 256

        # compute loss
        # dist(hypo - path)
        # https://stackoverflow.com/questions/76979995/pytorch-mse-loss-differs-from-direct-calculation-by-factor-of-2
        #losses = F.pairwise_distance(rep_hypo, rep_path, p=2).square().mean()   # mean over batch
        losses = F.mse_loss(rep_hypo, rep_path, reduction='sum') / rep_hypo.shape[0]  # mean over batch  # equivalently
        
        return losses



class ContrastiveRegLoss(nn.Module):
    """
    This class implements triplet loss. Given a triplet of (anchor, positive, negative),
    the loss minimizes the distance between anchor and positive while it maximizes the distance
    between anchor and negative. It compute the following loss function:

    loss = max(||anchor - positive|| - ||anchor - negative|| + margin, 0).

    Margin is an important hyperparameter and needs to be tuned respectively.

    For further details, see: https://en.wikipedia.org/wiki/Triplet_loss

    :param model: SentenceTransformerModel
    :param distance_metric: Function to compute distance between two embeddings. The class TripletDistanceMetric contains common distance metrices that can be used.
    :param triplet_margin: The negative should be at least this much further away from the anchor than the positive.

    Example::

        from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
        from sentence_transformers.readers import InputExample

        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        train_examples = [InputExample(texts=['Anchor 1', 'Positive 1', 'Negative 1']),
            InputExample(texts=['Anchor 2', 'Positive 2', 'Negative 2'])]
        train_dataset = SentencesDataset(train_examples, model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.TripletLoss(model=model)
    """
    def __init__(self, model: SentenceTransformer, triplet_margin: float = 1, batch=16):
        super(ContrastiveRegLoss, self).__init__()
        self.model = model
        self.triplet_margin = triplet_margin
        self.batch = batch
        #self.loss = CompositionalLoss(model)

    def get_config_dict(self):
        return {}

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        # organize input and pass to compositional loss

        # 1 hypo matrix + 32 path matrices + 32 negative path matrices
        #assert len(sentence_features) == 2 * self.batch + 1

        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]

        batch_size = reps[0].shape[0]

        rep_hypo = reps[0]  # batch * 256
        pos_reps = reps[1:(batch_size+1)] # batch
        neg_reps = reps[(batch_size+1):]  # batch

        # sum each path and stack paths
        rep_pos = torch.stack([torch.sum(rep_pos, dim=0) for rep_pos in pos_reps], dim=0)   # batch * 256
        rep_neg = torch.stack([torch.sum(rep_neg, dim=0) for rep_neg in neg_reps], dim=0)   # batch * 256

        # compute loss
        # max(||anchor - positive|| - ||anchor - negative|| + margin, 0)
        
        #pos_dist = F.pairwise_distance(rep_hypo, rep_pos, p=2).square()
        #neg_dist = F.pairwise_distance(rep_hypo, rep_neg, p=2).square()
        pos_dist = F.mse_loss(rep_hypo, rep_pos, reduction='none').sum(dim=1)
        neg_dist = F.mse_loss(rep_hypo, rep_neg, reduction='none').sum(dim=1)

        losses = F.relu(pos_dist - neg_dist + self.triplet_margin)
        
        return losses.mean() * 100
    



# ---------- InfoNCE ----------

# https://github.com/RElbers/info-nce-pytorch
# explicitly paired, supervised
loss = InfoNCE(negative_mode='paired')
batch_size, num_negative, embedding_size = 32, 20, 768
query = torch.randn(batch_size, embedding_size)
positive_key = torch.randn(batch_size, embedding_size)
negative_keys = torch.randn(batch_size, num_negative, embedding_size)
output = loss(query, positive_key, negative_keys)


