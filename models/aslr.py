"""
This file contains the definition of our model based on
@ https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch/notebook
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import warnings
import numpy as np

warnings.simplefilter("ignore")


def make_trg_mask(trg):
    """
    Computes mask of decoder sequence
    :param trg: Dimension of target mask.
    :return: mask.
    """
    batch_size, trg_len = trg.shape
    # returns the lower triangular part of matrix filled with ones
    trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
        batch_size, 1, trg_len, trg_len
    )
    return trg_mask


class Embedding(nn.Module):
    """
    Word embedding
    """

    def __init__(self, vocab_size, embed_dim, device):
        """
        Initial constructor
        :param vocab_size: size of vocabulary.
        :param embed_dim: dimension of embeddings.
        """
        super(Embedding, self).__init__()
        self.device = device
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        """
        Computes embedding
        :param x: Word indices.
        :return: Emdedded words.
        """
        x = x.to(self.device)
        out = self.embed(x)
        return out.to(self.device)


class PostionalEncoding(nn.Module):
    """
    Words and Video positional encoding class.
    """

    def __init__(self, max_seq_len, embed_model_dim):
        """
        Initial constructor
        :param max_seq_len: length of input sequence.
        :param embed_model_dim: dimension of embedding.
        """
        super(PostionalEncoding, self).__init__()
        self.embed_dim = embed_model_dim

        pe = torch.zeros(max_seq_len, self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.embed_dim)))
                pe[pos, i + 1] = math.cos(
                    pos / (10000 ** ((2 * (i + 1)) / self.embed_dim))
                )
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Computes positional encoding
        :param x: Word or Frame.
        :return: positioned Word of Frame.
        """

        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:, :seq_len], requires_grad=False)
        return x


class SpatialEncoding(nn.Module):
    """
    Frame embedding
    """

    def __init__(self, max_seq_len, embed_model_dim, device):
        """
        Initial Cunstructor
        :param max_seq_len: length of input sequence.
        :param embed_model_dim: dimension of embedding.
        """
        super(SpatialEncoding, self).__init__()
        self.embed_dim = embed_model_dim
        self.device = device
        pe = torch.zeros(max_seq_len, self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.embed_dim)))
                pe[pos, i + 1] = math.cos(
                    pos / (10000 ** ((2 * (i + 1)) / self.embed_dim))
                )
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Computes spatial embedding of a video frame
        :param x: feature extracted from frame. 1 D array.
        :return: X in the new embedding.
        """

        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        # add constant to embedding
        seq_len = x.size(0)
        x = x + torch.autograd.Variable(self.pe[:, :seq_len], requires_grad=False)
        return x.to(self.device)


class MultiHeadAttention(nn.Module):
    """
    Attention layer class
    """

    def __init__(self, embed_dim, n_heads, device):
        """
        Initial constructor.
        :param embed_dim: dimension of embedding vector output.
        :param n_heads: number of self attention heads.
        """
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.device = device
        self.single_head_dim = int(self.embed_dim / self.n_heads)

        # key,query and value matrixes    #64 x 64
        self.query_matrix = nn.Linear(
            self.single_head_dim, self.single_head_dim, bias=False, dtype=float
        )  # single key matrix for all 8 keys #512x512
        self.key_matrix = nn.Linear(
            self.single_head_dim, self.single_head_dim, bias=False, dtype=float
        )
        self.value_matrix = nn.Linear(
            self.single_head_dim, self.single_head_dim, bias=False, dtype=float
        )
        self.out = nn.Linear(
            self.n_heads * self.single_head_dim, self.embed_dim, dtype=float
        )

    def forward(self, key, query, value, mask=None):
        """
        Attention computation.
        :param key: key vector.
        :param query: query vector.
        :param value: value vector.
        :param mask: mask for decoder.
        :return: Computed attention.
        """
        key, query, value = (
            key.to(self.device),
            query.to(self.device),
            value.to(self.device),
        )
        batch_size = key.size(0)
        seq_length = key.size(1)
        # 32x10x512
        key = key.view(
            batch_size, seq_length, self.n_heads, self.single_head_dim
        )  # batch_size x sequence_length x n_heads x single_head_dim = (32x10x8x64)
        # key = key.to(dtype=torch.double)
        query = query.view(
            batch_size, seq_length, self.n_heads, self.single_head_dim
        )  # (32x10x8x64)
        value = value.view(
            batch_size, seq_length, self.n_heads, self.single_head_dim
        )  # (32x10x8x64)
        k = self.key_matrix(key)  # (32x10x8x64)
        q = self.query_matrix(query)
        v = self.value_matrix(value)

        q = q.transpose(
            1, 2
        )  # (batch_size, n_heads, seq_len, single_head_dim)    # (32 x 8 x 10 x 64)
        k = k.transpose(1, 2)  # (batch_size, n_heads, seq_len, single_head_dim)
        v = v.transpose(1, 2)  # (batch_size, n_heads, seq_len, single_head_dim)

        # computes attention
        # adjust key for matrix multiplication
        k_adjusted = k.transpose(
            -1, -2
        )  # (batch_size, n_heads, single_head_dim, seq_ken)  #(32 x 8 x 64 x 10)
        product = torch.matmul(
            q, k_adjusted
        )  # (32 x 8 x 10 x 64) x (32 x 8 x 64 x 10) = #(32x8x10x10)

        if mask is not None:
            mask = mask.to(self.device)
            product = product.masked_fill(mask == 0, float("-1e20"))

        # divising by square root of key dimension
        product = product / math.sqrt(self.single_head_dim)  # / sqrt(64)

        # applying softmax
        scores = F.softmax(product, dim=-1)

        # mutiply with value matrix
        scores = torch.matmul(
            scores, v
        )  ##(32x8x 10x 10) x (32 x 8 x 10 x 64) = (32 x 8 x 10 x 64)

        # concatenated output
        concat = (
            scores.transpose(1, 2)
                .contiguous()
                .view(batch_size, seq_length, self.single_head_dim * self.n_heads)
        )  # (32x8x10x64) -> (32x10x8x64)  -> (32,10,512)

        output = self.out(concat)  # (32,10,512) -> (32,10,512)

        return output.to(self.device)


class TransformerBlock(nn.Module):
    """
    Transformer block
    """

    def __init__(self, device, embed_dim, expansion_factor, n_heads, dropout_rate):
        """
        Initial constructor
        :param embed_dim: dimension of the embedding.
        :param expansion_factor: factor which determines output dimension of linear layer.
        :param n_heads: number of attention heads.
        """
        self.embed_dim = embed_dim
        self.expansion_factor = expansion_factor
        self.n_heads = n_heads
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(self.embed_dim, self.n_heads, device)

        self.norm1 = nn.LayerNorm(self.embed_dim, dtype=float)
        self.norm2 = nn.LayerNorm(self.embed_dim, dtype=float)

        self.feed_forward = nn.Sequential(
            nn.Linear(
                self.embed_dim, self.expansion_factor * self.embed_dim, dtype=float
            ),
            nn.SiLU(),
            nn.Linear(
                self.expansion_factor * self.embed_dim,
                self.expansion_factor * self.embed_dim,
                dtype=float,
            ),
            nn.SiLU(),
            nn.Linear(
                self.expansion_factor * self.embed_dim, self.embed_dim, dtype=float
            ),
        )

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, key, query, value, mask=None):
        """
        Performs forward pass in attention network
        :param key: key vector.
        :param query: query vector.
        :param value: value vector.
        :param mask:  mask to be given for multi head attention(used only for the decoder).
        :return: output of transformer block
        """

        attention_out = self.attention(key, query, value, mask)  # 32x10x512
        attention_residual_out = attention_out + value  # 32x10x512
        norm1_out = self.dropout1(self.norm1(attention_residual_out))  # 32x10x512

        feed_fwd_out = self.feed_forward(
            norm1_out
        )  # 32x10x512 -> #32x10x2048 -> 32x10x512
        feed_fwd_residual_out = feed_fwd_out + norm1_out  # 32x10x512
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out))  # 32x10x512

        return norm2_out


class TransformerEncoder(nn.Module):
    """
    Encoder of the transformer architecture.
    """

    def __init__(
            self,
            device,
            seq_len,
            embed_dim,
            num_layers,
            expansion_factor,
            n_heads,
            dropout_rate,
    ):
        """
        Initial constructor
        :param seq_len: length of input sequence.
        :param embed_dim: dimension of embedding.
        :param num_layers: number of encoder layers.
        :param expansion_factor: factor which determines number of linear layers in feed forward layer.
        :param n_heads: number of heads in multi-head attention.
        """
        super(TransformerEncoder, self).__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.expansion_factor = expansion_factor
        self.embedding_layer = Embedding(self.seq_len, self.embed_dim, device)
        # self.embedding_layer = Embedding(vocab_size, self.embed_dim)
        # self.positional_encoder = PostionalEncoding(seq_len, self.embed_dim)
        self.positional_encoder = SpatialEncoding(self.seq_len, self.embed_dim, device)
        self.dropout_rate = dropout_rate
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    device,
                    self.embed_dim,
                    self.expansion_factor,
                    self.n_heads,
                    self.dropout_rate,
                )
                for i in range(self.num_layers)
            ]
        )

    def forward(self, x):
        """
        Encodes frames
        :param x: Feature vector of a frame
        :return: encoded frame
        """
        positions = np.arange(start=0, stop=self.seq_len, step=1, dtype=int)
        embed_out = self.embedding_layer(torch.IntTensor(positions))
        # embed_out = self.embedding_layer(x)
        out = self.positional_encoder(embed_out)
        out = x + out
        for layer in self.layers:
            out = layer(out, out, out)

        return out


class DecoderBlock(nn.Module):
    """
    Transformer block
    """

    def __init__(self, device, embed_dim, expansion_factor, n_heads, dropout_rate):
        """
        Initial constructor
        :param embed_dim: dimension of the embedding.
        :param expansion_factor: factor which determines output dimension of linear layer.
        :param n_heads: number of attention heads.
        """
        super(DecoderBlock, self).__init__()
        self.embed_dim = embed_dim
        self.expansion_factor = expansion_factor
        self.n_heads = n_heads
        self.attention = MultiHeadAttention(self.embed_dim, self.n_heads, device)
        self.norm = nn.LayerNorm(self.embed_dim, dtype=float)
        self.dropout = nn.Dropout(dropout_rate)
        self.transformer_block = TransformerBlock(
            device, self.embed_dim, self.expansion_factor, self.n_heads, dropout_rate
        )

    def forward(self, key, query, x, mask):
        """
        Decodes words
        :param key: key vector.
        :param query: query vector.
        :param x: input word
        :param mask: mask to be given for multi head attention.
        :return: output of transformer block.
        """
        x = x.to(dtype=float)
        mask = mask.to(dtype=float)
        attention = self.attention(x, x, x, mask)  # 32x10x512
        value = self.dropout(self.norm(attention + x))
        out = self.transformer_block(key, query, value, mask)

        return out


class TransformerDecoder(nn.Module):
    """
    Decoder of the transformer architecture.
    """

    def __init__(
            self,
            device,
            target_vocab_size,
            embed_dim,
            seq_len,
            num_layers,
            expansion_factor,
            n_heads,
            dropout_rate,
    ):
        """
        Initial constructor
        :param target_vocab_size: vocabulary size of target.
        :param embed_dim: dimension of embedding.
        :param seq_len: length of input sequence.
        :param num_layers: number of decoder layers.
        :param expansion_factor: factor which determines number of linear layers in feed forward layer.
        :param n_heads: number of heads in multihead attention.
        :param device: Device to use. CUDA or GPU.
        :param dropout_rate: probability to apply dropout.
        """
        super(TransformerDecoder, self).__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.expansion_factor = expansion_factor
        self.target_vocab_size = target_vocab_size
        self.word_embedding = Embedding(self.target_vocab_size, self.embed_dim, device)
        self.position_embedding = PostionalEncoding(self.seq_len, self.embed_dim)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    device,
                    self.embed_dim,
                    expansion_factor=self.expansion_factor,
                    n_heads=self.n_heads,
                    dropout_rate=dropout_rate,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.fc_out = nn.Linear(self.embed_dim, self.target_vocab_size, dtype=float)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_out, trg_mask):
        """
        Decodes The words
        :param x: input vector from target.
        :param enc_out: output from encoder layer.
        :param trg_mask: mask for decoder self attention.
        :return: output vector
        """
        batch_size, seq_length = x.shape[0], x.shape[1]
        # batch_size, seq_length = 1, len(x)

        x = self.word_embedding(x)  # 32x10x512
        x = self.position_embedding(x)  # 32x10x512
        x = self.dropout(x)
        x = x.double()

        for layer in self.layers:
            x = layer(enc_out, enc_out, x, trg_mask)

        out = self.fc_out(x)

        return out


class Transformer(nn.Module):
    """
    Transformer model
    """

    def __init__(
            self,
            device,
            embed_dim,
            src_vocab_size,
            target_vocab_size,
            seq_length_en,
            seq_length_dec,
            num_layers,
            expansion_factor,
            n_heads,
            dropout_rate,
    ):
        """
        Initial constructor
        :param embed_dim: dimension of embedding.
        :param src_vocab_size: vocabulary size of source.
        :param target_vocab_size: vocabulary size of target.
        :param seq_length_en: length of input sequence at encoder.
        :param seq_length_dec: length of input sequence at decoder.
        :param num_layers: number of encoder layers.
        :param expansion_factor: factor which determines number of linear layers in feed forward layer.
        :param n_heads:  number of heads in multihead attention.
        :param device: Use GPU or CPU.
        :param dropout_rate: probability to apply dropout.
        """
        super(Transformer, self).__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.src_vocab_size = src_vocab_size
        self.target_vocab_size = target_vocab_size
        self.seq_length_en = seq_length_en
        self.seq_length_dec = seq_length_dec
        self.num_layers = num_layers
        self.expansion_factor = expansion_factor
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.encoder = TransformerEncoder(
            self.device,
            self.seq_length_en,
            self.embed_dim,
            self.num_layers,
            self.expansion_factor,
            self.n_heads,
            self.dropout_rate,
        )
        self.decoder = TransformerDecoder(
            self.device,
            self.target_vocab_size,
            self.embed_dim,
            self.seq_length_dec,
            self.num_layers,
            self.expansion_factor,
            self.n_heads,
            self.dropout_rate,
        )

    def forward(self, src, trg):
        """
        Performs the whole computations of the model
        :param src: input sequence.
        :param trg: target sequence.
        :return: final vector which returns probabilities of each target word
        """
        trg_mask = make_trg_mask(trg)
        enc_src = self.encoder(src)
        out = self.decoder(trg, enc_src, trg_mask)
        return out.to(self.device)
