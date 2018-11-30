# -*- coding: utf-8 -*-
"""
Created on 2018/11/25

@author: Gong

Reference: https://github.com/bojone/seq2seq/blob/master/seq2seq.py
Reference: https://kexue.fm/archives/5861
"""

import os

import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import Callback
from keras.layers import Input, Layer, Embedding, Bidirectional, CuDNNLSTM, Dense, Lambda, Activation
from keras.models import Model, load_model
from keras.optimizers import Adam

from loader import load_config, generate_config

TEST = 0

MODEL_FILE = "seq2seq_model.h5"
WEIGHTS_FILE = "model.weights"

CONFIG_FILE = "config.json"
MIN_COUNT = 32

MAX_LEN = 400
ID_MASK = 0
ID_NUK = 1
ID_BEG = 2
ID_EOS = 3

WORD_SIZE = 64
MAX_TITLE_LEN = 50

BATCH_SIZE = 16
EPOCHS = 50
STEPS_PER_EPOCH = 1000 # ~ 5 mins

def load_vocab(input_file):
    if not os.path.exists(CONFIG_FILE):
        generate_config(input_file=input_file,
                        config_file=CONFIG_FILE, min_count=MIN_COUNT)
    global vocab, id_word, word_id
    vocab, id_word, word_id = load_config(CONFIG_FILE)


def get_id(str, is_title=False):
    if is_title:
        idx = [word_id.get(word, 1) for word in str[:MAX_LEN - 2]]
        idx = [ID_BEG] + idx + [ID_EOS]
    else:
        idx = [word_id.get(word, 1) for word in str[:MAX_LEN]]
    return idx


def get_str(idx):
    return "".join([id_word.get(i, "") for i in idx])


def padding(x):
    max_len = max([len(i) for i in x])
    return [i + [0] * (max_len - len(i)) for i in x]


def data_generator():
    X, Y = [], []
    while True:
        df = pd.read_csv(input_file)
        for index, row in df.iterrows():
            if (not isinstance(row["title"], str)) or (not isinstance(row["content"], str)):
                continue
            X.append(get_id(row["content"]))
            Y.append(get_id(row["title"], is_title=True))
            if len(X) == BATCH_SIZE:
                X = np.array(padding(X))
                Y = np.array(padding(Y))
                yield [X, Y], None
                X, Y = [], []


def to_one_hot(x):
    x, x_mask = x
    x = K.cast(x, "int32")
    x = K.one_hot(x, len(vocab) + 4)
    x = K.sum(x_mask * x, 1, keepdims=True)
    x = K.cast(K.greater(x, 0.5), "float32")
    return x


class ScaleShift(Layer):

    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)

    def build(self, input_shape):
        kernel_shape = (1,) * (len(input_shape) - 1) + (input_shape[-1],)
        self.log_scale = self.add_weight(name='log_scale',
                                         shape=kernel_shape,
                                         initializer='zeros')
        self.shift = self.add_weight(name='shift',
                                     shape=kernel_shape,
                                     initializer='zeros')

    def call(self, inputs):
        x_outs = K.exp(self.log_scale) * inputs + self.shift
        return x_outs


class Interact(Layer):

    def __init__(self, **kwargs):
        super(Interact, self).__init__(**kwargs)

    def build(self, input_shape):
        in_dim = input_shape[0][-1]
        out_dim = input_shape[1][-1]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(in_dim, out_dim),
                                      initializer='glorot_normal')

    def call(self, inputs):
        q, v, v_mask = inputs
        k = v
        mv = K.max(v - (1. - v_mask) * 1e10, axis=1,
                   keepdims=True)
        mv = mv + K.zeros_like(q[:, :, :1])
        qw = K.dot(q, self.kernel)
        a = K.batch_dot(qw, k, [2, 2]) / 10.
        a -= (1. - K.permute_dimensions(v_mask, [0, 2, 1])) * 1e10
        a = K.softmax(a)
        o = K.batch_dot(a, v, [2, 1])
        return K.concatenate([o, q, mv], 2)

    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][1],
                input_shape[0][2] + input_shape[1][2] * 2)


def gen_title(content, top_k=3):
    xid = np.array([get_id(content)] * top_k)
    yid = np.array([[ID_BEG]] * top_k)
    scores = [0] * top_k
    for i in range(MAX_TITLE_LEN):
        prob = model.predict([xid, yid])[:, i, 3:]
        log_prob = np.log(prob + 1e-6)
        arg_top_k = log_prob.argsort(axis=1)[:, -top_k:]
        _yid = []
        _scores = []
        if i == 0:
            for j in range(top_k):
                _yid.append(list(yid[j]) + [arg_top_k[0][j] + 3])
                _scores.append(scores[j] + log_prob[0][arg_top_k[0][j]])
        else:
            for j in range(len(xid)):
                for k in range(top_k):
                    _yid.append(list(yid[j]) + [arg_top_k[j][k] + 3])
                    _scores.append(scores[j] + log_prob[j][arg_top_k[j][k]])
            _arg_top_k = np.argsort(_scores)[-top_k:]
            _yid = [_yid[k] for k in _arg_top_k]
            _scores = [_scores[k] for k in _arg_top_k]
        yid = []
        scores = []
        for k in range(len(xid)):
            if _yid[k][-1] == ID_EOS:
                return get_str(_yid[k])
            else:
                yid.append(_yid[k])
                scores.append(_scores[k])
        yid = np.array(yid)
    return get_str(yid[np.argmax(scores)])


class Evaluate(Callback):
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # s1 = u'夏天来临，皮肤在强烈紫外线的照射下，晒伤不可避免，因此，晒后及时修复显得尤为重要，否则可能会造成长期伤害。专家表示，选择晒后护肤品要慎重，芦荟凝胶是最安全，有效的一种选择，晒伤严重者，还请及时就医。'
        # s2 = u'8月28日，网络爆料称，华住集团旗下连锁酒店用户数据疑似发生泄露。从卖家发布的内容看，数据包含华住旗下汉庭、禧玥、桔子、宜必思等10余个品牌酒店的住客信息。泄露的信息包括华住官网注册资料、酒店入住登记的身份信息及酒店开房记录，住客姓名、手机号、邮箱、身份证号、登录账号密码等。卖家对这个约5亿条数据打包出售。第三方安全平台威胁猎人对信息出售者提供的三万条数据进行验证，认为数据真实性非常高。当天下午，华住集团发声明称，已在内部迅速开展核查，并第一时间报警。当晚，上海警方消息称，接到华住集团报案，警方已经介入调查。'
        # print(gen_title(s1))
        # print(gen_title(s2))
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights(WEIGHTS_FILE)


def seq2seq_model():
    x_in = Input(shape=(None,))
    y_in = Input(shape=(None,))
    x = x_in
    y = y_in
    x_mask = Lambda(lambda x: K.cast(
        K.greater(K.expand_dims(x, 2), 0), "float32"))(x)
    y_mask = Lambda(lambda x: K.cast(
        K.greater(K.expand_dims(x, 2), 0), "float32"))(y)

    x_one_hot = Lambda(to_one_hot)([x, x_mask])
    x_prior = ScaleShift()(x_one_hot)

    embedding = Embedding(len(vocab) + 4, WORD_SIZE)
    x = embedding(x)
    y = embedding(y)

    x = Bidirectional(CuDNNLSTM(WORD_SIZE // 2, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(WORD_SIZE // 2, return_sequences=True))(x)

    y = CuDNNLSTM(WORD_SIZE, return_sequences=True)(y)
    y = CuDNNLSTM(WORD_SIZE, return_sequences=True)(y)

    xy = Interact()([y, x, x_mask])
    xy = Dense(512, activation='relu')(xy)
    xy = Dense(len(vocab) + 4)(xy)
    xy = Lambda(lambda x: (x[0] + x[1]) / 2)([xy, x_prior])
    xy = Activation('softmax')(xy)

    cross_entropy = K.sparse_categorical_crossentropy(y_in[:, 1:], xy[:, :-1])
    loss = K.sum(cross_entropy * y_mask[:, 1:, 0]) / K.sum(y_mask[:, 1:, 0])

    global model
    model = Model([x_in, y_in], xy)
    model.add_loss(loss)
    model.compile(optimizer=Adam(1e-3))
    evaluator = Evaluate()
    model.fit_generator(data_generator(),
                        steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=EPOCHS,
                        callbacks=[evaluator])
    model.save(MODEL_FILE)
    return model


def predict(input_file, output_file):
    model.load_weights(WEIGHTS_FILE)
    df = pd.read_csv(input_file)
    content = []
    title = []
    for index, row in df.iterrows():
        if (not isinstance(row["title"], str)) or (not isinstance(row["content"], str)):
            continue
        title.append(row["title"])
        content.append(row["content"])
    result = {"actual": [], "predict": []}
    for i in range(len(content)):
        result["actual"].append(title[i])
        result["predict"].append(gen_title(content[i]))
    df = pd.DataFrame(result)
    df.to_csv(output_file)


if __name__ == "__main__":
    if TEST == 1:
        input_file = r"data\news_test.csv"
    else:
        input_file = r"data\news_train.csv"
    load_vocab(input_file)
    print("Vocab loaded.")
    model = seq2seq_model()
    predict(r"data\news_test.csv", "result_1.csv")
    print("DONE.")
