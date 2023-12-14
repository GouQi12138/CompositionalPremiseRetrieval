# changing sbert's data collection function
def raw_batching_collate(self, batch):
    text_list = [[]]
    labels = []

    queries = []

    for example in batch:
        queries.append(example["query"])
        text_list.append(example["positive"])
        labels.append(0)
    text_list[0] = queries

    if 'negative' in batch[0]:
        for example in batch:
            text_list.append(example["negative"])
            labels.append(1)

    labels = torch.tensor(labels)

    features = []
    for texts in text_list:
        tokenized = self.tokenize(texts)
        features.append(tokenized)

    return features, labels
