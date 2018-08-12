import torch
import torch.nn as nn
import torch.nn.functional as F

class KimCNN(nn.Module):
    def __init__(self, config):
        super(KimCNN, self).__init__()
        output_channel = config.output_channel
        target_class = config.target_class
        words_num = config.words_num
        words_dim = config.words_dim
        embed_num = config.embed_num
        embed_dim = config.embed_dim
        self.mode = config.mode
        Ks = 3 # There are three conv net here
        if config.mode == 'multichannel':
            input_channel = 2
        else:
            input_channel = 1

        self.conv1 = nn.Conv2d(input_channel, output_channel, (3, words_dim), padding=(2,0))
        self.conv2 = nn.Conv2d(input_channel, output_channel, (4, words_dim), padding=(3,0))
        self.conv3 = nn.Conv2d(input_channel, output_channel, (5, words_dim), padding=(4,0))

        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(Ks * output_channel, target_class)


    def forward(self, x):
        x = [F.relu(self.conv1(x)).squeeze(3), F.relu(self.conv2(x)).squeeze(3), F.relu(self.conv3(x)).squeeze(3)]
        # (batch, channel_output, ~=sent_len) * Ks
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # max-over-time pooling
        # (batch, channel_output) * Ks
        x = torch.cat(x, 1) # (batch, channel_output * Ks)
        x = self.dropout(x)
        logit = self.fc1(x) # (batch, target_size)
        return logit
