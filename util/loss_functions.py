

from info_nce import InfoNCE, info_nce


# https://github.com/RElbers/info-nce-pytorch
# explicitly paired, supervised
loss = InfoNCE(negative_mode='paired')
batch_size, num_negative, embedding_size = 32, 20, 768
query = torch.randn(batch_size, embedding_size)
positive_key = torch.randn(batch_size, embedding_size)
negative_keys = torch.randn(batch_size, num_negative, embedding_size)
output = loss(query, positive_key, negative_keys)


