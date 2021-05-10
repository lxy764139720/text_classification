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
        true_positives += K.sum(K.round(K.clip(y_true * y_pred, 0, 1))).numpy()
        possible_positives += K.sum(K.round(K.clip(y_true, 0, 1))).numpy()
        predicted_positives += K.sum(K.round(K.clip(y_pred, 0, 1))).numpy()
        total += len(y_true)
    accuracy = right / total
    recall = true_positives / (possible_positives + K.epsilon())
    precision = true_positives / (predicted_positives + K.epsilon())
    f1_score = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return accuracy, recall, precision, f1_score


class Evaluator(keras.callbacks.Callback):
    """评估与保存"""
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc, val_recall, val_precision, val_f1 = evaluate(test_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
        print(u'val_acc: %.5f, best_val_acc: %.5f\n' % (val_acc, self.best_val_acc))
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


def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads ** 2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数


# 写好函数后，启用对抗训练只需要一行代码
adversarial_training(model, 'Embedding-Token', 0.5)

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
