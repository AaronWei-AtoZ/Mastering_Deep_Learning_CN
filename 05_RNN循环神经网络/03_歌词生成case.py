"""
基于杰伦歌词来训练模型，用给定的起始词，结合长度，来进行AI歌词生成

实现步骤：
    1. 获取数据 进行分词 获取词表
    2. 数据预处理，构建数据集
    3. 搭建RNN神经网络
    4. 模型训练
    5. 模型预测
"""
import torch
import jieba
from torch.utils.data import DataLoader
import torch.nn as nn
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data' / 'jaychou_lyrics.txt'
MODEL_PATH = BASE_DIR / 'model' / 'text_generator.pth'

# 1. 获取数据 进行分词 获取词表
def build_vocab():
    # 记录去重后所有的词 每行文本分词结果
    unique_words, all_words = [], []
    # 遍历数据集，获得每行文本
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f'未找到语料：{DATA_PATH}。请阅读同目录 DATA_SOURCE.md，并使用自有或获授权文本。'
        )

    with DATA_PATH.open('r', encoding='utf-8') as corpus_file:
        for line in corpus_file:
            # 获取到每行歌词，进行分词
            words = jieba.lcut(line)
            # 所有分词结果记录到 all_words当中
            all_words.append(words)
            # 遍历分词结果，去重后添加到unique_words中
            for word in words:
                if word not in unique_words:
                    unique_words.append(word)
    # 统计语料中 去重后 词的数量
    unique_word_count = len(unique_words)  # 5703
    # 构建字典形式词表 key = 词，value = 词索引
    word2index = {word:i for i, word in enumerate(unique_words)} # {'想要': 0, '有': 1, '直升机': 2, '\n': 3, '和': 4, '你': 5, '飞到':6 ...}
    # 歌词文本用词表索引表示
    corpus_idx = []
    # 遍历每一行的分词结果
    for words in all_words:
        # 记录词索引列表
        tmp = []
        # 获取每行的 1词 及其 索引
        for word in words:
            tmp.append(word2index[word])
        # 在每行词之间，添加空格隔开，这样能知道什么时候一句话结束（原始数据采用空格作为标记）
        if ' ' in word2index:
            tmp.append(word2index[' '])
        # 获取文档中每个词的索引，添加到corpus_idx中
        corpus_idx.extend(tmp)
    # 返回结果：唯一词列表、词表、去重后词的数量、歌词文本用词表索引表示
    return unique_words, word2index, unique_word_count, corpus_idx

# 2. 数据预处理，构建数据集
# 定义数据集类，继承自 torch.utils.data.Dataset
class LyricsDataset(torch.utils.data.Dataset):
    # 初始化词索引、词个数等
    def __init__(self, corpus_idx, num_chars):
        # 文档数据中词的索引
        self.corpus_idx = corpus_idx
        # 每个句子中词的个数 自己设定的多少个词为一句！！！
        self.num_chars = num_chars
        # 文档数据中词的总数量，不去重版
        self.word_count = len(self.corpus_idx)
        # 训练的句子数量                                       # 注意注意注意！！！这里不需要真的是一句话为一个句子 只需要往后拿就行 因为这样才能知道上下文的关联
        self.number = (self.word_count - 1) // self.num_chars

    # 当使用len（obj）的时候，会自动调用此方法
    def __len__(self):
        return self.number

    # 当使用obj[index]时，自动调用
    def __getitem__(self, idx):
        # idx 指的是词的索引，并将其修正为索引值 到 文档的范围里边
        # 确保索引start在合法范围内
        start = idx * self.num_chars
        # 计算结束索引
        end = start + self.num_chars
        # 输入值，从文档中取出 start ~ end 的索引的词，作为x(上一轮的输入)
        x = self.corpus_idx[start:end]          # 不包右
        # 输出值，即网络预测结果y(当前时刻的输出)
        y = self.corpus_idx[start+1:end+1]
        # 返回张量形式的输入和输出
        return torch.tensor(x), torch.tensor(y)

# 3. 搭建RNN神经网络
class Model(nn.Module):
    def __init__(self, unique_word_count):      # unique_word_count 去重的词的数量（5703）
        super().__init__()
        # 搭建词嵌入层, 需要 词的数量、词向量的维度（人为规定）
        self.embed = nn.Embedding(unique_word_count, 128)
        # 循环网络层, 需要 词向量的维度，隐藏层的维度，网络层数
        self.rnn = nn.RNN(128, 256, 1)
        # 输出层，全连接层 需要 特征向量维度（和隐藏向量维度一致），词表中词的个数
        self.out = nn.Linear(256, unique_word_count)

    def forward(self, inputs, hidden):
        # 初始化词嵌入层处理
        embed = self.embed(inputs)
        # RNN处理
        output, hidden = self.rnn(embed.transpose(0, 1), hidden)
        # 全连接层，输入内容必须是二维数据，词的数量 * 词的维度
        output = self.out(output.reshape(shape=(-1, output.shape[-1])))
        # 返回预测结果 和 隐藏层
        return output, hidden

    # 隐藏层的初始化方法
    def init_hidden(self, bs):      # batch_size
        # 隐藏层的初始化:[网络层数，batch，隐藏层的向量维度]
        return torch.zeros(1, bs, 256)

# 4. 模型训练
def train():
    # 构建词典
    unique_words, word2index, unique_word_count, corpus_idx = build_vocab()
    # 获取数据集
    lyrics = LyricsDataset(corpus_idx, 32)
    # 初始化模型
    model = Model(len(unique_words))
    # 创建数据加载器
    data_loader = DataLoader(lyrics, batch_size=5, shuffle=True)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # 模型训练
    epochs = 10
    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss, iteration = 0.0, 0
        model.train()
        # 一轮下 各批次的训练动作
        # 直接遍历数据集，后台会调用LyricsDataset中的__getitem__()函数，获取到每个样本的数据和标签
        for x, y in data_loader:
            # 获取隐藏层初始值
            hidden = model.init_hidden(x.size(0))
            # 7.6 模型计算
            output, hidden = model(x, hidden)
            # 7.7 计算损失
            # y的形状：(batch，seq_len, 词向量维度) -> 转成一维向量 -> 每个词的下标索引
            # output的形状：(seq_len, batch, 词向量维度)
            y = torch.transpose(y, 0, 1).reshape(shape=(-1,))
            loss = criterion(output, y )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            iteration += 1

        print(f'epoch:{epoch +1}, time:{time.time() - start_time:.2f}s, loss:{epoch_loss / iteration:.4f}')

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)

# 5. 模型预测
def evaluate(start_word, sentence_len):
    # 构建词典
    unique_words, word2index, unique_word_count, corpus_idx = build_vocab()
    # 获取模型
    model = Model(len(unique_words))
    # 加载模型参数
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'未找到模型权重：{MODEL_PATH}。请先运行 train()。')
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True))
    model.eval()
    # 获取隐藏层初始化
    hidden = model.init_hidden(1)
    # 将输入词作为索引
    if start_word not in word2index:
        raise ValueError(f'起始词“{start_word}”不在当前词表中。')
    word_idx = word2index[start_word]
    # 定义列表，存放：产生的词的索引
    generate_sentence = [word_idx]
    # 遍历句子长度，获取到每一个词
    with torch.no_grad():
        for _ in range(sentence_len):
            # 模型预测
            output, hidden = model(torch.tensor([[word_idx]]), hidden)
            # 获取预测结果
            word_idx = int(torch.argmax(output).item())
            # 把预测结果添加到列表中
            generate_sentence.append(word_idx)

    # 将索引转为词 并打印                !!!找词只能去 unique_words里面找，而不是word2vec里面，因为word2vec是字典，不能根据值来找键
    for idx in generate_sentence:
        print(unique_words[idx], end='')


if __name__ == '__main__':
    # 1.获取数据 进行分词 获取词表
    # unique_words, word2index, unique_word_count, corpus_idx  = build_vocab()
    # print(f'词的数量:{word_count}')
    # print(f'去重后的词:{unique_words}')
    # print(f'词表:{word2index}')
    # print(f'用索引来表示文档:{corpus_idx}')

    # 2.构建数据集
    # dataset = LyricsDataset(corpus_idx, 5)
    # print(f'句子数量:{len(dataset)}')
    # 查看输入值 和 目标值
    # x, y = dataset[0]
    # print(x, y)

    # 创建模型对象
    # model = Model(len(unique_words))

    # 训练并保存模型
    # train()

    # 测试模型：仓库不包含语料和模型权重，请先准备语料并训练。
    if DATA_PATH.exists() and MODEL_PATH.exists():
        evaluate("星星", 50)
    else:
        print('未检测到语料或模型权重。请阅读 DATA_SOURCE.md，并先运行 train()。')
