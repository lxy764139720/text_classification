import os
import numpy as np
from bert4keras.backend import keras, search_layer, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Lambda, Dense
from keras.utils import to_categorical

num_classes = 3
maxlen = 128
batch_size = 32

# BERT base
config_path = './chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './chinese_L-12_H-768_A-12/vocab.txt'

texts = []
labels_index = {u'健康': 0, u'教育': 1, u'财经': 2}
labels = []
TEXT_PATH = '../preprocessing'
for name in os.listdir(TEXT_PATH):
    if name.split('.')[-1] == 'txt':
        class_name = name.split('.')[0]
        fpath = os.path.join(TEXT_PATH, name)
        with open(fpath, encoding='utf-8') as f:
            for l in f.readlines():
                texts.append(l.split(' '))
                labels.append(labels_index[class_name])
        print(fpath)
        print(class_name)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
train_data = [(X_train[i], y_train[i]) for i in range(len(y_train))]
test_data = [(X_test[i], y_test[i]) for i in range(len(y_test))]

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器"""

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size)
test_generator = data_generator(test_data, batch_size)

from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras import backend as K


def evaluate(data):
    total, right, true_positives, possible_positives, predicted_positives = 0., 0., 0., 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        right += (y_true == y_pred).sum()
        true_positives += K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives += K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives += K.sum(K.round(K.clip(y_pred, 0, 1)))
        total += len(y_true)
    accuracy = right / total
    recall = true_positives / (possible_positives + K.epsilon())
    precision = true_positives / (predicted_positives + K.epsilon())
    f1_score = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return accuracy, recall, precision, f1_score


class Evaluator(keras.callbacks.Callback):
    """评估与保存"""

    def on_epoch_end(self, epoch, logs=None):
        val_acc, val_recall, val_precision, val_f1 = evaluate(test_generator)
        print(u'val_acc: %.5f\n' % val_acc)
        print(u'val_recall: %.5f\n' % val_recall)
        print(u'val_precision: %.5f\n' % val_precision)
        print(u'val_f1: %.5f\n' % val_f1)


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0])(bert.model.output)
output = Dense(num_classes, activation='softmax', kernel_initializer=bert.initializer)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),
    metrics=['sparse_categorical_accuracy'],
)

evaluator = Evaluator()

my_callbacks = [
    EarlyStopping(patience=5),
    TensorBoard(log_dir='./logs'),
    evaluator,
    ModelCheckpoint('BERT.h5', monitor='val_acc', save_best_only=True, mode='auto'),
]

model.fit(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=50,
    callbacks=my_callbacks
)
