

def create_vocab_dict(text_file):
    with open(text_file,'r') as f:
        lines = f.readlines()
    lines = [line.strip().split() for line in lines]

    w2i = {'PAD': 0,
           'UNK': 1}

    i2w = {0: 'PAD',
           1: 'UNK'}

    idx = 2
    for line in lines:
        for tok in line:
          if not tok in w2i:
              w2i[tok] = idx
              i2w[idx] = tok
              idx += 1
    return w2i,i2w

        
    
