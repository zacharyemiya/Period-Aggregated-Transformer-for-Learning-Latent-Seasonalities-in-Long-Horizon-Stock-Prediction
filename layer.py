import torch
from torch import nn
from einops import repeat
import math
import torch.nn.functional as F


class TwoAttentionLayer(nn.Module):
    def __init__(self, n_heads, d_model, factor, dropout=0.1):
        super().__init__()
        self.season_attention = AttentionLayer(n_heads, d_model)
        self.season_receive = AttentionLayer(n_heads, d_model)
        self.trend_attention = TrendAttentionLayer(n_heads, d_model)
        self.trend_sender = TrendAttentionLayer(n_heads, d_model)
        self.season_inter_trend = AttentionLayer(n_heads, d_model)
        self.router = nn.Parameter(torch.rand(factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model))

    def forward(self, x_s, x_t):
        batch = x_t.shape[0]
        x_s_enc = self.season_attention(x_s, x_s, x_s)[0] # B, L, d_model
        x_t_enc = self.trend_attention(x_t, x_t, x_t)[0] # B, L, d_model
        x_enc = self.season_inter_trend(x_s, x_s, x_s, x_t)[0]
        #
        trend_in = x_t_enc + self.dropout(x_t_enc)
        trend_in = self.norm1(trend_in)
        season_in = x_s_enc + self.dropout(x_s_enc)
        season_in = self.norm2(season_in)
        #
        batch_router = repeat(self.router, 'f d -> b f d', b=batch)
        trend_buffer = self.trend_sender(batch_router, trend_in, trend_in)[0]
        season_receive = self.season_receive(season_in, trend_buffer, trend_buffer)[0]
        dim_in = x_enc + self.dropout(x_enc)
        dim_in = self.norm3(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm4(dim_in)
        final_out = dim_in + self.dropout(self.MLP2(season_receive))
        return final_out


class AttentionLayer(nn.Module):
    def __init__(self, n_heads, d_model, factor=1):
        super(AttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.factor = factor

        d_keys = d_model // n_heads
        d_values = d_model // n_heads

        self.query = nn.Linear(d_model, d_keys * n_heads)
        self.key = nn.Linear(d_model, d_keys * n_heads)
        self.value = nn.Linear(d_model, d_values * n_heads)
        self.out = nn.Linear(d_values * n_heads, d_model)

        self.season_attention = SeasonAttention(factor)
        self.trend_attention = TrendAttentionLayer(n_heads, d_model)

    def seasonplus(self, queries, keys, values, trend_attns=None):
        B, L, H, E = queries.shape
        _, S, _, _ = keys.shape

        # topk = int(math.log(L) * self.factor)
        topk = 3
        tmp_values = values
        delays_agg = torch.zeros_like(values.permute(0, 2, 3, 1).contiguous()).float()
        _,_, corr = self.season_attention(queries, keys, values)
        for i in range(topk):
            season_map, index, _ = self.season_attention(queries, keys, values)  # B, topk
            pattern = torch.roll(tmp_values, -int(index[i]), -1).permute(0, 2, 3, 1).contiguous() #B,H,E,S
            if trend_attns is not None:
                delays_agg = delays_agg + pattern * \
                             (season_map[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, H, E, L)) \
                            + torch.einsum('bhls, bhds->bhds', trend_attns[i], pattern)
            else:
                delays_agg = delays_agg + pattern * \
                             (season_map[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, H, E, L))
        return delays_agg, corr

    def forward(self, q, k, v, x_t=None):
        B, L, _ = q.shape
        _, S, _ = k.shape
        H = self.n_heads

        queries = self.query(q).reshape(B, L, H, -1)
        keys = self.key(k).reshape(B, S, H, -1)
        values = self.value(v).reshape(B, S, H, -1)
        if x_t is not None:
            trend_value, trend_attns = self.trend_attention(x_t, x_t, x_t)
            season_plus_trend, corr = self.seasonplus(queries, keys, values, trend_attns)

            out = season_plus_trend.view(B, L, -1)
        else:
            season_plus_trend , corr= self.seasonplus(queries, keys, values)
            out = season_plus_trend.view(B, L, -1)

        return out, corr


class SeasonAttention(nn.Module):

    def __init__(self, factor=1):
        super().__init__()
        self.factor = factor

    def corr_weights(self, values, corr):
        batch, length, head, channel = values.shape
        topk = int(math.log(length) * self.factor)+4
        mean_value = torch.mean(torch.mean(corr, dim=1),dim=1) #B,L
        index = torch.topk(torch.mean(mean_value, dim=0), topk, dim=-1)[1] #topk个值
        weights = torch.stack([mean_value[:, index[i]] for i in range(topk)], dim=-1) # B, topk
        corr_weights = torch.softmax(weights, dim=-1) #归一化
        return corr_weights, index

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1) #q_fft: B, H, E, L
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, n=L, dim=-1) #B,H,E,L
        corr_weights, index = self.corr_weights(values, corr) #B, topk
        return corr_weights, index, corr


class TrendAttentionLayer(nn.Module):

    def __init__(self, n_heads, d_model, factor=1, dropout=0.1):
        super().__init__()

        d_keys = d_model // n_heads
        d_values = d_model // n_heads

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.inner_attention = TrendAttention(iter=3, attention_dropout = dropout)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn_lists = self.inner_attention(queries, keys, values)

        return out, attn_lists


class TrendAttention(nn.Module):
    def __init__(self, iter, attention_dropout=0.1):
        super().__init__()
        self.iter = iter
        self.attention_dropout = attention_dropout

    def trend_interaction(self, queries, keys):
        length = queries.shape[1]
        attn_list = []
        for i in range(self.iter):
            indexes = torch.arange(0, length, i+1)# i=2, [0,2,4,6,8,10,12,14,16,18];i=3, [0,3,6,9,12,15,18]
            tmp_q = queries[:, indexes, :, :] #B，L，H, D
            tmp_attn = torch.einsum('blhe,bshe->bhls', tmp_q, keys)
            attn_list.append(tmp_attn)
        return attn_list

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        attn_list = self.trend_interaction(queries, keys)
        values_list = []
        for att in attn_list:
            att = torch.softmax(att, dim=-1)
            att = F.dropout(att, p=self.attention_dropout, training=self.training)
            tmp_values = torch.einsum('bhls,bshe->bshe', att, values) #B, S, H, D
            values_list.append(tmp_values)

        final_values = torch.mean(torch.stack(values_list, dim=1), dim=1)
        out = final_values.view(B, S, -1)
        return out, attn_list


if __name__ == '__main__':
    x_s = torch.randn(32, 64, 8)
    x_t = torch.randn(32, 64, 8)

    ttention = TwoAttentionLayer(4, 8, 7)
    a = ttention(x_s, x_t)

















