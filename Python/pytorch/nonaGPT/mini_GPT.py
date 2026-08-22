import torch
import urllib.request
import torch.nn as nn
import torch.nn.functional as F   #作为 torch.nn 的函数式接口，里面有很多常用的函数，比如 softmax、relu 等
import torch.optim as optim    #优化器

# ===== 0. 配置区：所有超参数集中管理 =====
block_size = 64      # 上下文窗口：模型能看多远（64）
n_embd = 128         # 向量维度（128）
n_head = 4           # 注意力头数
n_layer = 4          # Transformer 层数（4）
batch_size = 64      # 每批样本数（64）
dropout = 0.2        # 训练时随机关闭 20% 连接，防止死记硬背
# ===== 1. 数据准备 =====
# tiny shakespeare：莎士比亚全部作品，约 1MB，nanoGPT 的经典入门数据集
text = open("input.txt", encoding="utf-8").read()
chars = sorted(list(set(text)))   # 所有不重复字符 → 这就是字符级"词表"
vocab_size = len(chars)           # 词表大小


# ===== 2. 分词器：字符 <-> 整数 双向翻译 =====
stoi = {ch: i for i, ch in enumerate(chars)}      # 字符 → ID
itos = {i: ch for i, ch in enumerate(chars)}      # ID → 字符
encode = lambda s: [stoi[c] for c in s]           # "abc" → [a的id, b的id, c的id]
decode = lambda l: ''.join(itos[i] for i in l)    # [id...] → "abc"


# # 验证：往返必须还原原句
# #sample = "To be or not to be, that is the question"
# sample = "hello"
# ids = encode(sample)
# print(f"encode: {sample!r} -> {ids}")
# print(f"decode: {ids} -> {decode(ids)!r}")

# ===== 3. 数据切分：整本书 → 训练/验证集 =====

data = torch.tensor(encode(text), dtype=torch.long)   # 整本书变成一个 ID 张量
n = int(0.9 * len(data))          # 前 90% 训练，后 10% 验证
train_data, val_data = data[:n], data[n:]

# 看一个长度 block_size 的训练样本长什么样：
# block_size = 8                    # 模型一次最多"看"几个字符
# x = train_data[:block_size]       # 输入：前 8 个字符的 ID
# y = train_data[1:block_size+1]    # 目标：右移一位，逐位置预测下一个字符
# for t in range(block_size):
#     ctx, nxt = x[:t+1].tolist(), y[t].item()
#     print(f"看过的文本 {decode(ctx)!r:20} -> 要预测 {decode([nxt])!r}")

# ===== 4. 模型（第一步）：词嵌入 + 位置嵌入 =====
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer): #vocab_size: 词表大小, n_layer: block 数量, n_head: 注意力头数量, n_embd: 词向量维度, block_size: 上下文长度
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)      # 字符 → 向量
        self.position_embedding = nn.Embedding(block_size, n_embd)   # 位置 → 向量
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        )                                # n_layer 个 block 串联
        self.ln_f = nn.LayerNorm(n_embd) # 输出前最后归一化                           # 新增：逐位思考
        self.lm_head = nn.Linear(n_embd, vocab_size)     # 新增：输出 logits
    def forward(self, idx):
        B, T = idx.shape                             # 解包批次和长度
        assert T <= self.block_size, f"输入长度 {T} 超过 block_size {self.block_size}"   #
        tok = self.token_embedding(idx)              # (B, T, n_embd) 词向量
        pos = self.position_embedding(torch.arange(T))  # (T, n_embd) 位置向量，arange(T)为[0, 1, 2, ..., T-1]
        x = tok + pos                                # 词义和位置叠加
        x = self.blocks(x)               # 多层 block 串联
        x = self.ln_f(x)                 # 最后归一化
        logits = self.lm_head(x)
        return logits
    
# ===== 5. 单头因果自注意力 =====
class Head(nn.Module):
    """单头自注意力：每个 token 看前面所有 token，加权混合信息"""
    def __init__(self, n_embd, head_size, block_size):  #self, n_embd: 词向量维度, head_size: 注意力头维度, block_size: 上下文长度
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)  # 书脊标签
        self.query = nn.Linear(n_embd, head_size, bias=False)  # 我想找什么
        self.value = nn.Linear(n_embd, head_size, bias=False)  # 找到的书内容
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))  # 因果掩码
        self.dropout = nn.Dropout(dropout)      # 注意力权重上随机抹掉 20%

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1)              # (B, T, T) 注意力分数
        wei = wei * (C ** -0.5)                    # scaled：防止 softmax 饱和
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # 因果：未来设为 -inf
        wei = F.softmax(wei, dim=-1)               # 归一化成权重
        wei = self.dropout(wei)                    # 在权重上随机抹掉 20%（防"背答案"）
        out = wei @ v                              # (B, T, head_size) 加权求和
        return out
    
# ===== 6. 多头注意力：多个头各看各的，最后拼接 =====
class MultiHeadAttention(nn.Module):
    """n_head 个单头并行工作，输出拼回 n_embd 维"""
    def __init__(self, n_embd, n_head, head_size, block_size): #self, n_embd: 词向量维度, n_head: 注意力头数量, head_size: 注意力头维度, block_size: 上下文长度
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size) for _ in range(n_head)]
        )

    def forward(self, x):
        # 每个头独立计算，结果在最后一维拼接 → (B, T, n_head*head_size)
        return torch.cat([h(x) for h in self.heads], dim=-1)

# ===== 7. MLP：逐 token 独立思考 =====
class MLP(nn.Module):
    """每个 token 独立地对自己的向量做非线性变换"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # 先拓宽到 4 倍，装下更多中间想法
            nn.ReLU(),                      # 非线性：没有它，两层线性=一层线性
            nn.Linear(4 * n_embd, n_embd),  # 再压缩回 n_embd，形状守恒
        )
        self.dropout = nn.Dropout(dropout)      # 全连接层同样防背诵
    def forward(self, x):
        x = self.net(x)    # (B, T, n_embd) -> (B, T, n_embd)
        return self.dropout(x)
# ===== 8. Block：注意力 + MLP + 残差 + LayerNorm（transformer 基本单元） =====
class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)                              # 注意力前归一化
        self.attn = MultiHeadAttention(n_embd, n_head, n_embd // n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)                              # MLP 前归一化
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # 归一化 → 注意力 → 残差相加
        x = x + self.mlp(self.ln2(x))    # 归一化 → MLP → 残差相加
        return x
# ===== 9. 训练循环 =====

# 取一批随机数据：从训练集随机切 block_size 长的片段
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))   # 随机起点
    x = torch.stack([data[i:i+block_size] for i in ix])         # 输入块
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])     # 目标：右移一位
    return x, y

batch_size = 64
model = TinyGPT(vocab_size, block_size, n_embd, n_head, n_layer)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)   # AdamW 比 SGD 稳

for step in range(600):
    xb, yb = get_batch("train")                        # ① 抽一批
    logits = model(xb)                                 # 前向预测
    B, T, C = logits.shape
    loss = F.cross_entropy(logits.view(B*T, C), yb.view(B*T))  # ② 算 loss
    optimizer.zero_grad()
    loss.backward()                                    # ③ 反向传播
    optimizer.step()                                   # ④ 更新参数

    if step % 100 == 0:                                # 每 100 步打印一次
        xv, yv = get_batch("val")
        logits_v = model(xv)
        loss_v = F.cross_entropy(logits_v.view(B*T, C), yv.view(B*T))
        print(f"step {step:4d} | train loss {loss.item():.4f} | val loss {loss_v.item():.4f}")

# ===== 10. 文本生成 =====
@torch.no_grad()                          # 推理模式：不需要梯度
def generate(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]   # 只看最后 block_size 个字符（记忆上限！）
        logits = model(idx_cond)          # (1, T, vocab)
        logits = logits[:, -1, :]         # 只要最后一个位置的预测
        probs = F.softmax(logits, dim=-1) # 转成概率
        idx_next = torch.multinomial(probs, num_samples=1)  # 按概率抽样
        idx = torch.cat((idx, idx_next), dim=1)  # 新字符拼进序列
    return idx

# 用训练好的模型生成一段"莎士比亚"
start = torch.tensor([encode("First Citizen:")], dtype=torch.long)
out = generate(model, start, max_new_tokens=200)
print(decode(out[0].tolist()))