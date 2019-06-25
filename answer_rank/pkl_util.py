import pickle


def load_pkl(path):
    with open(path, 'rb') as fr:
        x = pickle.load(fr)
        return x


def to_pkl(x, save_path):
    with open(save_path, 'wb') as fw:
        pickle.dump(x, fw)