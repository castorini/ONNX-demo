from torchtext import data
import os


class SST1Dataset(data.TabularDataset):
    dirname = 'data'
    @classmethod
    def splits(cls, text_field, label_field,
               train='phrases.train', validation='dev', test='test'):
        prefix_name = 'stsa.fine.'
        path = '/data/datasets/SST/'
        return super(SST1Dataset, cls).splits(
            path, train=prefix_name + train, validation=prefix_name + validation, test=prefix_name + test,
            format='tsv', fields=[('label', label_field), ('text', text_field)]
        )
