from tqdm import tqdm
import array
import torch
import six
import sys

fname_txt = sys.argv[1]
fname_pt = sys.argv[2]
itos, vectors, dim = [], array.array('d'), None
with open(fname_txt, 'rb') as f:
    lines = [line for line in f]
print("Loading vectors from {}".format(fname_txt))
for line in tqdm(lines, total=len(lines)):
    entries = line.strip().split(b' ')
    word, entries = entries[0], entries[1:]
    if dim is None:
        dim = len(entries)
    try:
        if isinstance(word, six.binary_type):
            word = word.decode('utf-8')
    except:
        print('non-UTF8 token', repr(word), 'ignored')
        continue
    vectors.extend(float(x) for x in entries)
    itos.append(word)

stoi = {word: i for i, word in enumerate(itos)}
vectors = torch.Tensor(vectors).view(-1, dim)
print('saving vectors to', fname_pt)
torch.save((stoi, vectors, dim), fname_pt)
