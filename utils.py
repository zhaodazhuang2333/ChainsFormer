import torch
import gzip
import torch.nn as nn
import numpy as np
import pdb
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print('set seed:', seed)
    return None
def r2_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_mean = np.mean(y_true)
    ss_tot = np.sum(np.square(y_true - y_mean))
    ss_res = np.sum(np.square(y_true - y_pred))
    r2 = 1 - ss_res / ss_tot
    return r2

def path_data_read(raw_path, dataset):
    file_path = raw_path + dataset + '/one_hop_query_with_path.gz'
    with gzip.open(file_path, 'rb') as f:
        one_hop_path = torch.load(f)
    file_path = raw_path + dataset + '/two_hop_query_with_path.gz'
    with gzip.open(file_path, 'rb') as f:
        two_hop_path = torch.load(f)
    file_path = raw_path + dataset + '/three_hop_query_with_path.gz'
    with gzip.open(file_path, 'rb') as f:
        three_hop_path = torch.load(f)
    return one_hop_path, two_hop_path, three_hop_path

def get_init_embedding(n_entity, n_relation, n_num_relation, hidden_dim, device='cuda'):
    gamma = 6.0
    epsilon = 1.0
    embedding_range = nn.Parameter(
                torch.Tensor([(gamma + epsilon) / hidden_dim]),
                requires_grad=False
            )
    entity_embedding = nn.Parameter(torch.zeros(n_entity, hidden_dim, device=device))
    nn.init.uniform_(
        tensor=entity_embedding,
        a=-embedding_range.item(),
        b=embedding_range.item()
    )
    relation_embedding = nn.Parameter(torch.zeros(n_relation, hidden_dim, device=device))
    nn.init.uniform_(
        tensor=relation_embedding,
        a=-embedding_range.item(),
        b=embedding_range.item()
    )
    num_relation_embedding = nn.Parameter(torch.zeros(n_num_relation, hidden_dim, device=device))
    nn.init.uniform_(
        tensor=num_relation_embedding,
        a=-embedding_range.item(),
        b=embedding_range.item()
    )
    reg_embedding = nn.Parameter(torch.zeros((1, hidden_dim, 1), device=device))
    missing_embedding = nn.Parameter(torch.zeros(hidden_dim, device=device))
    return entity_embedding, relation_embedding, num_relation_embedding, reg_embedding, missing_embedding


def get_path_embedding_witout_entity(path, entity_embedding, relation_embedding, num_relation_embedding, reg_embedding, hop=1):
    if hop == 1:
        path_embedding = torch.stack((num_relation_embedding[path[:, -2].astype(int)], relation_embedding[path[:, -4].astype(int)], num_relation_embedding[path[:, -6].astype(int)]), dim=-1)
        path_embedding = torch.cat((path_embedding, reg_embedding.expand(path_embedding.shape[0], -1, -1)), dim=-1)
    elif hop == 2:
        path_embedding = torch.stack((num_relation_embedding[path[:, -2].astype(int)], relation_embedding[path[:, -4].astype(int)], relation_embedding[path[:, -6].astype(int)], num_relation_embedding[path[:, -8]]), dim=-1)
        path_embedding = torch.cat((path_embedding, reg_embedding.expand(path_embedding.shape[0], -1, -1)), dim=-1)
    elif hop == 3:
        path_embedding = torch.stack((num_relation_embedding[path[:, -2].astype(int)], relation_embedding[path[:, -4].astype(int)], relation_embedding[path[:, -6].astype(int)], relation_embedding[path[:, -8].astype(int)], num_relation_embedding[path[:, -10]]), dim=-1)
        path_embedding = torch.cat((path_embedding, reg_embedding.expand(path_embedding.shape[0], -1, -1)), dim=-1)
    
    value = path[:, -1].astype(float)
    return path_embedding.permute(0, 2, 1), value

def pad_to_length(tensor, target_length, pad_value):
    current_length = tensor.size(1)
    if current_length < target_length:
        padding_shape = (target_length - current_length)
        pad_value = pad_value.expand(tensor.shape[0], padding_shape, -1)
        tensor = torch.cat((tensor, pad_value), dim=1)
    return tensor, current_length - 1

def hyperbolic_distance(x, y, c=1):
    c = c**0.5
    result = 2 * torch.atanh(c * torch.norm(mobius_addition(-x, y, c), dim=1))/c
    return result

def mobius_addition(x, y, c=1):
    alpha_xy = (1 + 2 *c * torch.matmul(x.unsqueeze(1), y.unsqueeze(2))).squeeze() + c * torch.norm(y, dim=1) ** 2
    beta_xy = 1 - c * torch.norm(x, dim=1) ** 2
    result = (alpha_xy.unsqueeze(1) * x + beta_xy.unsqueeze(1) * y)/ ((1 + 2 * c * torch.matmul(x.unsqueeze(1), y.unsqueeze(2))).squeeze() + c * torch.norm(y, dim=1) ** 2  * torch.norm(y, dim=1) ** 2 ).unsqueeze(-1)
    return result

def hyperbolic_pruning(all_query, index, entity_embedding, relation_embedding, num_relation_embedding, reg_embedding, missing_token_embedding, one_hop_path, two_hop_path, three_hop_path, one_hop_sample_num=64, two_hop_sample_num=128, three_hop_sample_num=128, sample_num = 30, topk = False, device='cuda'):
    for i in range(len(index)):
        query_label = all_query[index[i], -3]
        one_hop_path_for_query = one_hop_path[index[i]]
        two_hop_path_for_query = two_hop_path[index[i]]
        three_hop_path_for_query = three_hop_path[index[i]]
        ## TODO 采样策略需要调整
        if len(one_hop_path_for_query) <= one_hop_sample_num:
            one_hop_path_for_query = np.concatenate((query_label * np.ones((len(one_hop_path_for_query), 1)),one_hop_path_for_query), axis=1)
        else:
            random_indices = np.random.choice(len(one_hop_path_for_query), size=one_hop_sample_num, replace=True)
            one_hop_path_for_query = one_hop_path_for_query[random_indices]
            one_hop_path_for_query = np.concatenate((query_label * np.ones((one_hop_sample_num, 1)),one_hop_path_for_query), axis=1)
        if len(two_hop_path_for_query) <= two_hop_sample_num:
            two_hop_path_for_query = np.concatenate((query_label * np.ones((len(two_hop_path_for_query), 1)),two_hop_path_for_query), axis=1)
        else:
            random_indices = np.random.choice(len(two_hop_path_for_query), size=two_hop_sample_num, replace=True)
            two_hop_path_for_query = two_hop_path_for_query[random_indices]
            two_hop_path_for_query = np.concatenate((query_label * np.ones((two_hop_sample_num, 1)),two_hop_path_for_query), axis=1)
        if len(three_hop_path_for_query) <= three_hop_sample_num:
            three_hop_path_for_query = np.concatenate((query_label * np.ones((len(three_hop_path_for_query), 1)),three_hop_path_for_query), axis=1)
        else:
            random_indices = np.random.choice(len(three_hop_path_for_query), size=three_hop_sample_num, replace=True)
            three_hop_path_for_query = three_hop_path_for_query[random_indices]
            three_hop_path_for_query = np.concatenate((query_label * np.ones((three_hop_sample_num, 1)),three_hop_path_for_query), axis=1)
        if i == 0:
            one_hop_path_indices = index[i] * np.ones(one_hop_path_for_query.shape[0])
            one_hop_path_counts = i * np.ones(one_hop_path_for_query.shape[0])
            two_hop_path_indices = index[i] * np.ones(two_hop_path_for_query.shape[0])
            two_hop_path_counts = i * np.ones(two_hop_path_for_query.shape[0])
            three_hop_path_indices = index[i] * np.ones(three_hop_path_for_query.shape[0])
            three_hop_path_counts = i * np.ones(three_hop_path_for_query.shape[0])
            one_hop_path_for_querys = one_hop_path_for_query
            two_hop_path_for_querys = two_hop_path_for_query
            three_hop_path_for_querys = three_hop_path_for_query
        else:
            one_hop_path_for_querys = np.concatenate((one_hop_path_for_querys, one_hop_path_for_query), axis=0)
            two_hop_path_for_querys = np.concatenate((two_hop_path_for_querys, two_hop_path_for_query), axis=0)
            three_hop_path_for_querys = np.concatenate((three_hop_path_for_querys, three_hop_path_for_query), axis=0)
            one_hop_path_indices = np.concatenate((one_hop_path_indices, index[i] * np.ones(one_hop_path_for_query.shape[0])), axis=0)
            one_hop_path_counts = np.concatenate((one_hop_path_counts, i * np.ones(one_hop_path_for_query.shape[0])), axis=0)
            two_hop_path_indices = np.concatenate((two_hop_path_indices, index[i] * np.ones(two_hop_path_for_query.shape[0])), axis=0)
            two_hop_path_counts = np.concatenate((two_hop_path_counts, i * np.ones(two_hop_path_for_query.shape[0])), axis=0)
            three_hop_path_indices = np.concatenate((three_hop_path_indices, index[i] * np.ones(three_hop_path_for_query.shape[0])), axis=0)
            three_hop_path_counts = np.concatenate((three_hop_path_counts, i * np.ones(three_hop_path_for_query.shape[0])), axis=0)
    one_hop_path_embedding, value1 = get_path_embedding_witout_entity(one_hop_path_for_querys, entity_embedding, relation_embedding, num_relation_embedding, reg_embedding, hop=1)
    two_hop_path_embedding, value2 = get_path_embedding_witout_entity(two_hop_path_for_querys, entity_embedding, relation_embedding, num_relation_embedding, reg_embedding, hop=2)
    three_hop_path_embedding, value3 = get_path_embedding_witout_entity(three_hop_path_for_querys, entity_embedding, relation_embedding, num_relation_embedding, reg_embedding, hop=3)
    one_hop_value_label = one_hop_path_for_querys[:, -2]
    two_hop_value_label = two_hop_path_for_querys[:, -2]
    three_hop_value_label = three_hop_path_for_querys[:, -2]
    value_label = np.concatenate((one_hop_value_label, two_hop_value_label, three_hop_value_label), axis=0)
    path_counts = np.concatenate((one_hop_path_counts, two_hop_path_counts, three_hop_path_counts), axis=0)
    one_hop_path_embedding, one_hop_length = pad_to_length(one_hop_path_embedding, 6, missing_token_embedding)
    two_hop_path_embedding, two_hop_length = pad_to_length(two_hop_path_embedding, 6, missing_token_embedding)
    three_hop_path_embedding, three_hop_length = pad_to_length(three_hop_path_embedding, 6, missing_token_embedding)
    path_embedding = torch.cat((one_hop_path_embedding, two_hop_path_embedding, three_hop_path_embedding), dim=0)
    value = np.concatenate((value1, value2, value3), axis=0)
    value_bit = np.unpackbits(np.array(value).view(np.uint8)).reshape(-1, 64)
    reg_token_indices = torch.cat((torch.zeros(len(one_hop_path_counts)) + one_hop_length, torch.zeros(len(two_hop_path_counts)) + two_hop_length, torch.zeros(len(three_hop_path_counts)) + three_hop_length), dim=0)
    one_hop_path_only_relation = one_hop_path_for_querys[:, [0, 2, 4]]
    new_column = -1 * np.ones((len(one_hop_path_for_querys), 2))
    one_hop_path_only_relation = np.hstack((one_hop_path_only_relation, new_column))
    two_hop_path_only_relation = two_hop_path_for_querys[:, [0, 2, 4, 6]]
    new_column = -1 * np.ones((len(two_hop_path_for_querys), 1))
    two_hop_path_only_relation = np.hstack((two_hop_path_only_relation, new_column))
    three_hop_path_only_relation = three_hop_path_for_querys[:, [0, 2, 4, 6, 8]]
    paths = np.vstack((one_hop_path_only_relation, two_hop_path_only_relation, three_hop_path_only_relation))
    y = all_query[index, -2]
    y_label = all_query[index, -3]
    one_hop_relation_embedding = one_hop_path_embedding[:, 1, :]
    one_hop_hpyerbolic_path_embedding = one_hop_relation_embedding
    two_hop_hyperbolic_path_embedding = mobius_addition(two_hop_path_embedding[:, 1, :], two_hop_path_embedding[:, 2, :])
    three_hop_hyperbolic_path_embedding = mobius_addition(mobius_addition(three_hop_path_embedding[:, 1, :], three_hop_path_embedding[:, 2, :]), three_hop_path_embedding[:, 3, :])

    one_hop_path_mertic = hyperbolic_distance(one_hop_path_embedding[:, 0, :], one_hop_hpyerbolic_path_embedding)
    two_hop_path_mertic = hyperbolic_distance(two_hop_path_embedding[:, 0, :], two_hop_hyperbolic_path_embedding)
    three_hop_path_mertic = hyperbolic_distance(three_hop_path_embedding[:, 0, :], three_hop_hyperbolic_path_embedding)
    

    one_hop_path_score = -one_hop_path_mertic + torch.cosine_similarity(one_hop_path_embedding[:, 0,:], one_hop_path_embedding[:, 2,:], dim=1)
    two_hop_path_score = -two_hop_path_mertic + torch.cosine_similarity(two_hop_path_embedding[:, 0,:], two_hop_path_embedding[:, 3,:], dim=1)
    three_hop_path_score = -three_hop_path_mertic + torch.cosine_similarity(three_hop_path_embedding[:, 0,:], three_hop_path_embedding[:, 4,:], dim=1)
    path_score = torch.cat((one_hop_path_score, two_hop_path_score, three_hop_path_score), dim=0)
    path_score1 = torch.exp(path_score)
    # pdb.set_trace()
    # path_score1 = 1 / path_score
    count = 0
    for query_id in np.unique(path_counts):
        query_path_index = np.where(path_counts == query_id)[0]
        query_path_score = path_score1[query_path_index]
        query_path_score = query_path_score / query_path_score.sum()
        if topk:
            if sample_num <= len(query_path_score):
                selected_index = torch.topk(query_path_score, sample_num, largest=True)[1]
            else:
                selected_index = torch.multinomial(query_path_score, sample_num, replacement=True)
        else:
            selected_index1 = torch.topk(query_path_score, int(sample_num/2), largest=True)[1]
            selected_index2 = torch.randint(0, len(query_path_score), (sample_num - int(sample_num/2),)).to(device)
            selected_index = torch.cat((selected_index1, selected_index2), dim=0)
        selected_index = torch.tensor(query_path_index).to(device)[selected_index]
        if count == 0:
            selected_indices = selected_index
        else:
            selected_indices = torch.cat((selected_indices, selected_index), dim=0)
        count += 1
    path_embedding = path_embedding[selected_indices]
    path_counts = path_counts[selected_indices.cpu()]
    value = value[selected_indices.cpu()]
    value_bit = value_bit[selected_indices.cpu()]
    reg_token_indices = reg_token_indices[selected_indices.cpu()]
    paths = paths[selected_indices.cpu()]
    value_label = value_label[selected_indices.cpu()]
    return y, y_label.astype(int), path_embedding, path_counts, value, value_bit, reg_token_indices, paths, value_label