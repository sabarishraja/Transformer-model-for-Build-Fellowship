import torch

class CharacterTokenizer:
  def __init__(self, content):
    self.vocab = sorted(list(set(content)))
    self.char_to_idx = { ch:i for i,ch in enumerate(self.vocab) }
    self.idx_to_char = { i:ch for i,ch in enumerate(self.vocab) }

  def encode(self, xs):
    return [self.char_to_idx[x] for x in xs]

  def decode(self, xs):
    return ''.join([self.idx_to_char[x] for x in xs])

class Dataset:
  def __init__(self, content, context_size, batch_size, split_factor=0.9):
    self.context_size = context_size
    self.batch_size = batch_size
    self.data = content
    assert 0 < split_factor < 1
    n = int(len(self.data) * split_factor)
    self.train_data, self.val_data = self.data[:n], self.data[n:]

  def get_batch(self, split, device, y_shift=1):
    data = self.train_data if split == 'train' else self.val_data
    ix = torch.randint(len(data) - self.context_size - y_shift, (self.batch_size,))
    x = torch.stack([data[i:i+self.context_size] for i in ix])
    y = torch.stack([data[i+y_shift:i+self.context_size+y_shift] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
