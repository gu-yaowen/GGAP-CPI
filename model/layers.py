import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class ProteinEncoder(nn.Module):
    def __init__(self, args, node_dim=1280):
        super(ProteinEncoder, self).__init__()
        self.args = args
        # gcn_dims = [1280] + [fc_dim, 300, 300]
        gcn_dims = [node_dim] + [300]
        gcn_layers = [GCNConv(gcn_dims[i-1], gcn_dims[i], bias=True) for i in range(1, len(gcn_dims))]

        self.gcn = nn.ModuleList(gcn_layers)

        self.drop1 = nn.Dropout(p=self.args.dropout) #dp 0.2

    def forward(self, data, pertubed=False):
        x = data.x
        x = self.drop1(x)
        for idx, gcn_layer in enumerate(self.gcn):
            x = F.relu(gcn_layer(x, data.edge_index.long()))
            if pertubed:
                random_noise = torch.rand_like(x).to(x.device)
                x = x + torch.sign(x) * F.normalize(random_noise, dim=-1) * 0.1
        data.x = x
        num_graphs = data.num_graphs
        node_embeddings_list = []
        graph_embeddings_list = []
        for i in range(num_graphs):
            mask = data.batch == i
            graph_embeddings = data.x[mask]
            node_embeddings_list.append(graph_embeddings)
            graph_embeddings_list.append(torch.mean(graph_embeddings, dim=0))
        graph_embeddings_list = torch.stack(graph_embeddings_list, dim=0)
        return node_embeddings_list, graph_embeddings_list
    

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        #        self.gelu = GELU()
        self.relu = nn.ReLU(inplace=True)
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


# class MultiHeadCrossAttentionPooling(nn.Module):
#     def __init__(self, d_model, num_heads, dropout_rate=0.1, pooling='mean'):
#         super(MultiHeadCrossAttentionPooling, self).__init__()
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_k = d_model // num_heads

#         # self.self_attention_norm = nn.LayerNorm(d_model)
#         self.query_linear = nn.Linear(d_model, d_model)
#         self.key_linear = nn.Linear(d_model, d_model)
#         self.value_linear = nn.Linear(d_model, d_model)
#         self.dropout = nn.Dropout(dropout_rate)

#         self.ffn_norm = nn.LayerNorm(d_model)
#         self.out_linear = FeedForwardNetwork(d_model, d_model)
#         self.self_attention_dropout = nn.Dropout(dropout_rate)
#         self.pooling = pooling

#     def forward(self, query_list, key_list):
#         # initialize padding tensors
#         max_n = max([q.size(0) for q in query_list])
#         max_m = max([k.size(0) for k in key_list])
        
#         padded_queries = torch.zeros((len(query_list), max_n, self.d_model)).to(query_list[0].device)
#         padded_keys = torch.zeros((len(key_list), max_m, self.d_model)).to(query_list[0].device)
#         query_masks = torch.zeros(len(query_list), max_n, dtype=torch.bool).to(query_list[0].device)
#         key_masks = torch.zeros(len(key_list), max_m, dtype=torch.bool).to(query_list[0].device)
        
#         for i, (q, k) in enumerate(zip(query_list, key_list)):
#             padded_queries[i, :q.size(0), :] = q
#             padded_keys[i, :k.size(0), :] = k
#             query_masks[i, :q.size(0)] = True
#             key_masks[i, :k.size(0)] = True

#         # padded_queries = self.self_attention_norm(padded_queries)
#         # padded_keys = self.self_attention_norm(padded_keys)
#         # linear transformation
#         queries_transformed = self.query_linear(padded_queries).view(len(query_list), max_n, self.num_heads, self.d_k)
#         keys_transformed = self.key_linear(padded_keys).view(len(key_list), max_m, self.num_heads, self.d_k)
#         values_transformed = self.value_linear(padded_keys).view(len(key_list), max_m, self.num_heads, self.d_k)

#         queries_transformed = queries_transformed.transpose(1, 2)
#         keys_transformed = keys_transformed.transpose(1, 2)
#         values_transformed = values_transformed.transpose(1, 2)

#         # calculate attention
#         scores = torch.matmul(queries_transformed, keys_transformed.transpose(-2, -1))
#         scores = scores / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

#         # masking
#         query_masks = query_masks.unsqueeze(1).unsqueeze(3)
#         key_masks = key_masks.unsqueeze(1).unsqueeze(2)
#         mask = query_masks & key_masks
#         scores = scores.masked_fill(~mask, float(1e-8))

#         # Softmax
#         attention = F.softmax(scores, dim=-1)
#         attention = self.dropout(attention)

#         # aggregation
#         context1 = torch.matmul(attention, values_transformed).transpose(1, 2).contiguous()
#         context1 = context1.view(len(query_list), max_n, self.d_model)
#         # also calculate the query attention
#         context2 = torch.matmul(attention.transpose(3, 2), queries_transformed).transpose(1, 2).contiguous()
#         context2 = context2.view(len(key_list), max_m, self.d_model)
#         # context = self.ffn_norm(context)

#         output1 = self.out_linear(context1)
#         output2 = self.out_linear(context2)
#         # output = self.dropout(output)
#         # output = self.ffn_norm(output)
#         # remove padding
#         outputs1 = [output1[i, :query_list[i].size(0), :] for i in range(len(query_list))]
#         outputs2 = [output2[i, :key_list[i].size(0), :] for i in range(len(key_list))]

#         # pooling
#         if self.pooling == 'mean':
#             pooled_outputs = [torch.concat([torch.mean(mol_feat, dim=0),
#                                             torch.mean(prot_feat, dim=0)], dim=-1) 
#                                             for mol_feat, prot_feat in zip(outputs1, outputs2)]
#         elif self.pooling == 'max':
#             pooled_outputs = [torch.concat([torch.max(mol_feat, dim=0)[0],
#                                             torch.max(prot_feat, dim=0)[0]], dim=-1) 
#                                             for mol_feat, prot_feat in zip(outputs1, outputs2)]
#         else:
#             raise ValueError("Unsupported pooling type. Use 'mean' or 'max'.")

#         return torch.stack(pooled_outputs, axis=0), attention

class MultiHeadCrossAttentionPooling(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate=0.05, pooling='mean'):
        # dp 0.1
        super(MultiHeadCrossAttentionPooling, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # self.self_attention_norm = nn.LayerNorm(d_model)
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(d_model)
        self.out_linear = FeedForwardNetwork(d_model, d_model)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        self.pooling = pooling

    def forward(self, query_list, key_list):
        # initialize padding tensors
        max_n = max([q.size(0) for q in query_list])
        max_m = max([k.size(0) for k in key_list])
        
        padded_queries = torch.zeros((len(query_list), max_n, self.d_model)).to(query_list[0].device)
        padded_keys = torch.zeros((len(key_list), max_m, self.d_model)).to(query_list[0].device)
        query_masks = torch.zeros(len(query_list), max_n, dtype=torch.bool).to(query_list[0].device)
        key_masks = torch.zeros(len(key_list), max_m, dtype=torch.bool).to(query_list[0].device)
        
        for i, (q, k) in enumerate(zip(query_list, key_list)):
            padded_queries[i, :q.size(0), :] = q
            padded_keys[i, :k.size(0), :] = k
            query_masks[i, :q.size(0)] = True
            key_masks[i, :k.size(0)] = True

        # padded_queries = self.self_attention_norm(padded_queries)
        # padded_keys = self.self_attention_norm(padded_keys)
        # linear transformation
        queries_transformed = self.query_linear(padded_queries).view(len(query_list), max_n, self.num_heads, self.d_k)
        keys_transformed = self.key_linear(padded_keys).view(len(key_list), max_m, self.num_heads, self.d_k)
        values_transformed = self.value_linear(padded_keys).view(len(key_list), max_m, self.num_heads, self.d_k)

        queries_transformed = queries_transformed.transpose(1, 2)
        keys_transformed = keys_transformed.transpose(1, 2)
        values_transformed = values_transformed.transpose(1, 2)

        # calculate attention
        scores = torch.matmul(queries_transformed, keys_transformed.transpose(-2, -1))
        scores = scores / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        # masking
        query_masks = query_masks.unsqueeze(1).unsqueeze(3)
        key_masks = key_masks.unsqueeze(1).unsqueeze(2)
        mask = query_masks & key_masks
        scores = scores.masked_fill(~mask, float(1e-8))

        # Softmax
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # aggregation
        context = torch.matmul(attention, values_transformed).transpose(1, 2).contiguous()
        context = context.view(len(query_list), max_n, self.d_model)
        # context = self.ffn_norm(context)

        output = self.out_linear(context)
        # output = self.dropout(output)
        # output = self.ffn_norm(output)
        # remove padding
        outputs = [output[i, :query_list[i].size(0), :] for i in range(len(query_list))]

        # pooling
        if self.pooling == 'mean':
            pooled_outputs = [torch.mean(o, dim=0) for o in outputs]
        elif self.pooling == 'max':
            pooled_outputs = [torch.max(o, dim=0)[0] for o in outputs]
        else:
            raise ValueError("Unsupported pooling type. Use 'mean' or 'max'.")

        return torch.stack(pooled_outputs, axis=0), attention