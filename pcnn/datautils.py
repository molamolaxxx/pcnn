import numpy as np

def pos_constrain(val, max_val=498, min_val=0):
    """位置约束,用来限制单词的位置值
    """
    val = val + 60
    return min(max_val, max(min_val, val))

def load_vocab(filename):
    """加载词字典
    word------>word id
    """
    try:
        d = dict()
        with open(filename) as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx

    except Exception as err:
        print(err)
    return d

def get_sequences_length(sequences):
    """
    获得一个batch内每个句子的长度
    """
    sequence_length = []

    for seq in sequences:
        seq = list(seq)
        sequence_length += [len(seq)]

    return sequence_length

def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (word_idx, pos1, pos2, relation) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    word_batch, pos1_batch, pos2_batch, entpos_batch, y_batch = [], [], [], [], []

    for (word, pos1, pos2,entpos, y) in data:

        if len(y_batch) == minibatch_size:
            #记录每个句子的长度
            sequence_lengths = get_sequences_length(word_batch)
            #print(word_batch)
            #print(entpos_batch)
            assert len(entpos_batch) == len(sequence_lengths)
            pos_batch = list()
            for idx, i in enumerate(entpos_batch):
                a, b = i
                #(实体1位置,实体2位置,句子长度)
                pos_batch.append([a, b, sequence_lengths[idx]])
            #print(pos_batch)
            yield word_batch, pos1_batch, pos2_batch, pos_batch, y_batch
            word_batch, pos1_batch, pos2_batch, entpos_batch, y_batch = [], [], [], [], []

        word_batch   += [word]
        pos1_batch   += [pos1]
        pos2_batch   += [pos2]
        entpos_batch += [entpos]
        y_batch      += [y]

    if len(y_batch) != 0:
        sequence_lengths = get_sequences_length(word_batch)
        assert len(entpos_batch) == len(sequence_lengths)
        pos_batch = list()
        for idx, i in enumerate(entpos_batch):
            a, b = i
            pos_batch.append([a, b, sequence_lengths[idx]])
        yield word_batch, pos1_batch, pos2_batch, pos_batch, y_batch

def pad_sequences(sequences, pad_tok=0):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
        a list record original length of sequences

    """
    _sequence_padded = []
    max_length = max(map(lambda x : len(x), sequences))

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        _sequence_padded +=  [seq_]

    sequence_padded = np.asarray(_sequence_padded)
    #print(sequence_padded)
    return sequence_padded

def shuffle_data(filename):
    from sys import platform
    import subprocess
    import os.path
    '''打乱文件的顺序'''
    try:
        subprocess.run(["shuf", "-o", filename, filename])
        print("Shuffle {} successfully".format(filename))
    except Exception as e:
        print(e)
        print("Failed to shuffle datasets.")

def to_piece(data, pos, width=2):
    """将句子以实体为标志，width为前后宽度，切分成左中右三个部分
    """
    assert len(data) == len(pos)
    #assert np.asarray(pos, dtype=np.int32).shape[1] == 3
    num = len(data)
    left = []
    mid = []
    right = []
    for i in range(num):
        left.append(data[i][0:(pos[i][0] + width)])
        mid.append(data[i][max(0, (pos[i][0] - width)): min((pos[i][1] + width), (pos[i][2] - 1))])
        right.append(data[i][(pos[i][1] - width):(pos[i][2] - 1)])

    return left, mid, right

def get_processing_word(vocab_words=None, allow_unk=True, UNK = "<UNK>"):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("hello") = (12345)
                 = (word id)

    """
    def f(word):
        # get id of word
        if vocab_words is not None:
            if word in vocab_words.keys():
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNK]
                else:
                    raise Exception("Unknow key is not allowed. Check that "\
                                    "your vocab (tags?) is correct")
        return word
    return f

def to_bags(data):
    """将数据放入bag,每一个bag有共同的relation
    """
    word_batch, pos1_batch, pos2_batch, pos_batch, y_batch = data
    #总共有多少种关系
    relations = set(y_batch)
    #有多少个包
    num_bags = len(list(relations))
    #句子向量包
    word_bags = [[] for i in range(num_bags)]
    pos1_bags = [[] for i in range(num_bags)]
    pos2_bags = [[] for i in range(num_bags)]
    pos_bags  = [[] for i in range(num_bags)]
    #关系包
    y_bags    = [[] for i in range(num_bags)]
    for idx, i in enumerate(relations):
        for idy, j in enumerate(y_batch):
            if i == j:
                word_bags[idx].append(word_batch[idy])
                pos1_bags[idx].append(pos1_batch[idy])
                pos2_bags[idx].append(pos2_batch[idy])
                pos_bags[idx].append(pos_batch[idy])
                y_bags[idx].append(y_batch[idy])

    return word_bags, pos1_bags, pos2_bags, pos_bags, y_bags, num_bags