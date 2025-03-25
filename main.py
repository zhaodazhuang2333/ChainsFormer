import os



import torch
import pandas as pd
import gzip
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from models import *
from torch_geometric.utils import scatter
from utils import *
import struct 
import pdb
import itertools
import argparse
import logging
from datetime import datetime

def setup_logging(log_file):
    # Set up logging configuration
    logging.basicConfig(filename=log_file,
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    # Create a handler to output logs to console as well
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

def create_timestamped_dir(base_dir):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_dir = os.path.join(base_dir, timestamp)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def parse_args():
    parser = argparse.ArgumentParser(description="Run model.")
    parser.add_argument('--file_path', type=str, default='data/', help='File path')
    parser.add_argument('--dataset', type=str, default='FB15K', help='Dataset')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size')
    parser.add_argument('--epoches', type=int, default=1000, help='Epoches')
    parser.add_argument('--sample_num', type=int, default=256, help='Sample num')
    parser.add_argument('--one_hop_sample_num', type=int, default=256, help='One hop sample num')
    parser.add_argument('--two_hop_sample_num', type=int, default=1536, help='Two hop sample num')
    parser.add_argument('--three_hop_sample_num', type=int, default=256, help='Three hop sample num')
    parser.add_argument('--add_model', action='store_true', help='Add model')
    parser.add_argument('--score_by_distance', action='store_false', help='Score by distance')
    parser.add_argument('--topk', action='store_false', help='select by Topk')
    parser.add_argument('--do_test', action='store_false', help='Do test')
    parser.add_argument('--seed', type=int, default=None, help='Seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--max_patience', type=int, default=75, help='Max patience')
    parser.add_argument('--num_layers', type=int, default=2, help='Max length')
    parser.add_argument('--num_heads', type=int, default=4, help='Max length')
    parser.add_argument('--d_ff', type=int, default=1024, help='Max length')
    parser.add_argument('--results_dir', type=str, default='output', help='Log file')
    parser.add_argument('--tips', type=str, default='None', help='tips')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--loss', type=str, default='L1', help='loss')

    return parser.parse_args()
def main(args):
    results_dir = create_timestamped_dir(args.results_dir)
    log_file = os.path.join(results_dir, 'training.log')
    setup_logging(log_file)
    logging.info(f"Loading data from {args.file_path}")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"args: {args}")
    gpu_id = args.gpu
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    seed = args.seed
    if seed is not None:
        set_seed(int(seed))
    else:
        seed = np.random.randint(10000)
        set_seed(seed)

    dataset = args.dataset
    graph_path = args.file_path + dataset + '.pt'
    embedding_save_path = os.path.join(results_dir, dataset + '_embedding_'+str(seed)+'.pt')
    model_save_path = os.path.join(results_dir, dataset +'_model_'+str(seed)+'.pt')
    train_graph = torch.load(graph_path)

    one_hop_path, two_hop_path, three_hop_path = path_data_read(args.file_path+'path_dataset/', dataset)
    logging.info(f"Initializing model")
    n_entity = train_graph.num_nodes
    n_relation = torch.unique(train_graph.edge_type).shape[0]
    n_num_relation = len(np.unique(train_graph.numerical_attr[:, 1]))
    hidden_dim = args.hidden_dim  
    pattern_dim = n_num_relation  
    entity_embedding, relation_embedding, num_relation_embedding, reg_embedding, missing_token_embedding= get_init_embedding(n_entity, n_relation, n_num_relation, hidden_dim, device=device)
    all_query = train_graph.numerical_attr 
    min_max = []
    for i in range(n_num_relation):
        data1 = all_query[all_query[:, -3] == i]
        min_value = data1[:, -2].min()
        max_value = data1[:, -2].max()
        min_max.append([min_value, max_value])
    min_max  = np.array(min_max)
    batch_size = args.batch_size
    train_query = all_query[all_query[:, -1] == 0]
    val_query = all_query[all_query[:, -1] == 1]
    test_query = all_query[all_query[:, -1] == 2]
    one_hop_sample_num = args.one_hop_sample_num
    two_hop_sample_num = args.two_hop_sample_num
    three_hop_sample_num = args.three_hop_sample_num
    sample_num = args.sample_num
    all_indices = np.arange(len(all_query))

    train_indices = all_indices[0: len(train_query)]
    val_indices = all_indices[len(train_query): len(train_query) + len(val_query)]
    test_indices = all_indices[len(train_query) + len(val_query):]

    # pdb.set_trace()
    train_indices = np.random.permutation(train_indices)

    train_num_batches = (len(train_indices) + batch_size - 1) // batch_size
    val_num_batches = (len(val_indices) + batch_size - 1) // batch_size
    test_num_batches = (len(test_indices) + batch_size - 1) // batch_size

    epoches = args.epoches
    model = Numerator(num_layers = args.num_layers, d_model = hidden_dim, num_heads = args.num_heads, d_ff = args.d_ff, max_length = 10, dropout=args.dropout).to(device)
    
    all_params = list(model.parameters()) + [relation_embedding, num_relation_embedding, reg_embedding, missing_token_embedding]
    optimizer = optim.Adam(all_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.loss == 'L1':
        crierion = nn.L1Loss()
    elif args.loss == 'L2':
        crierion = nn.MSELoss() 
    best_val_loss = 1000000
    max_patience = args.max_patience
    patience = 0
    label_range = min_max[:, 1] - min_max[:, 0]
    min_max = torch.tensor(min_max).to(device).float()

    for epoch in range(epoches):
        train_loss = 0
        model.train()
        all_y_norm = []
        all_pred_norm = []
        all_y_label = []
        np.random.shuffle(train_indices)
        for batch_id in range(train_num_batches):
            start_id = batch_id * batch_size
            end_id = min((batch_id + 1) * batch_size, len(train_indices))
            index = train_indices[start_id:end_id]
            if args.topk:
                y, y_label, embedding, path_counts, value, value_bit, reg_token_indices, path, value_label = hyperbolic_pruning(all_query, index, entity_embedding, relation_embedding, num_relation_embedding, reg_embedding, missing_token_embedding, one_hop_path, two_hop_path, three_hop_path, sample_num = sample_num, one_hop_sample_num= one_hop_sample_num,two_hop_sample_num=two_hop_sample_num ,three_hop_sample_num=three_hop_sample_num, topk=True)
            else:
                y, y_label, embedding, path_counts, value, value_bit, reg_token_indices, path, value_label = hyperbolic_pruning(all_query, index, entity_embedding, relation_embedding, num_relation_embedding, reg_embedding, missing_token_embedding, one_hop_path, two_hop_path, three_hop_path, sample_num = sample_num, one_hop_sample_num= one_hop_sample_num,two_hop_sample_num=two_hop_sample_num ,three_hop_sample_num=three_hop_sample_num)
            if np.unique(path_counts).shape[0] != batch_size:
                continue
            value = torch.tensor(value, dtype=torch.float32).to(device)
            valu_bit = torch.tensor(value_bit, dtype=torch.float32).to(device)
            y = torch.tensor(y).float().to(device)
            y_label = torch.tensor(y_label).to(device)
            y_norm = (y - min_max[y_label, 0])/ (min_max[y_label, 1] - min_max[y_label, 0])
            value_ratio, weight = model(embedding, valu_bit, reg_token_indices.to(int).to(device), torch.tensor(path_counts).to(int).to(device))
            if args.add_model:
                pred_value = value_ratio[:, 0] + value
                pred_value = pred_value * value_ratio[:, 1]
            else:
                pred_value = value_ratio.squeeze(1) * value
            pred_value = pred_value * weight.squeeze(1)
            pred_value = scatter(pred_value, torch.tensor(path_counts).to(int).to(device), reduce='sum')
            pred_value_normed = (pred_value - min_max[y_label, 0])/ (min_max[y_label, 1] - min_max[y_label, 0])
            all_pred_norm.extend(pred_value_normed.cpu().detach().numpy())
            all_y_norm.extend(y_norm.cpu().detach().numpy())
            all_y_label.extend(y_label.cpu().detach().numpy())
            loss = crierion(pred_value_normed.float(), y_norm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        all_pred_norm = np.array(all_pred_norm)
        all_y_norm = np.array(all_y_norm)
        all_y_label = np.array(all_y_label)
        ranges = label_range[all_y_label]
        all_pred = all_pred_norm * ranges + min_max[all_y_label, 0].detach().cpu().numpy()
        all_y = all_y_norm * ranges + min_max[all_y_label, 0].detach().cpu().numpy()
        normed_mae_error = np.abs(all_pred_norm - all_y_norm).mean()
        normed_rmse_error = np.sqrt(np.square(all_pred_norm - all_y_norm).mean())

        mae_error = np.abs(all_pred - all_y)
        mse_error = np.square(all_pred - all_y)
        mertic_per_label = dict()
        for i in range(n_num_relation):
            mertic_per_label[i] = dict()
            mertic_per_label[i]['mae'] = mae_error[all_y_label == i].mean()
            mertic_per_label[i]['rmse'] = np.sqrt(mse_error[all_y_label == i].mean())
            mertic_per_label[i]['r2'] = r2_score(all_y[all_y_label == i], all_pred[all_y_label == i])

        train_mertic_per_label = mertic_per_label
        train_all_normed_mertic = {'mae': normed_mae_error, 'rmse': normed_rmse_error}
        train_loss /= train_num_batches
        
        val_loss = 0
        model.eval()
        all_y_norm = []
        all_pred_norm = []
        all_y_label = []
        with torch.no_grad():
            for batch_id in range(val_num_batches):
                start_id = batch_id * batch_size
                end_id = min((batch_id + 1) * batch_size, len(val_indices))
                index = val_indices[start_id:end_id]
                if args.topk:
                    y, y_label, embedding, path_counts, value, value_bit, reg_token_indices, path, value_label = hyperbolic_pruning(all_query, index, entity_embedding, relation_embedding, num_relation_embedding, reg_embedding, missing_token_embedding, one_hop_path, two_hop_path, three_hop_path, sample_num = sample_num, one_hop_sample_num= one_hop_sample_num,two_hop_sample_num=two_hop_sample_num ,three_hop_sample_num=three_hop_sample_num, topk=True)
                else:
                    y, y_label, embedding, path_counts, value, value_bit, reg_token_indices, path, value_label = hyperbolic_pruning(all_query, index, entity_embedding, relation_embedding, num_relation_embedding, reg_embedding, missing_token_embedding, one_hop_path, two_hop_path, three_hop_path, sample_num = sample_num, one_hop_sample_num= one_hop_sample_num,two_hop_sample_num=two_hop_sample_num ,three_hop_sample_num=three_hop_sample_num)
                value = torch.tensor(value, dtype=torch.float32).to(device)
                valu_bit = torch.tensor(value_bit, dtype=torch.float32).to(device)
                y = torch.tensor(y).float().to(device)
                y_label = torch.tensor(y_label).to(device)
                y_norm = (y - min_max[y_label, 0])/ (min_max[y_label, 1] - min_max[y_label, 0])
                value_ratio, weight = model(embedding, valu_bit, reg_token_indices.to(int).to(device), torch.tensor(path_counts).to(int).to(device))
                if args.add_model:
                    pred_value = value_ratio[:, 0] + value
                    pred_value = pred_value * value_ratio[:, 1]
                else:
                    pred_value = value_ratio.squeeze(1) * value
                pred_value = pred_value * weight.squeeze(1)
                pred_value = scatter(pred_value, torch.tensor(path_counts).to(int).to(device), reduce='sum')
                pred_value_normed = (pred_value - min_max[y_label, 0])/ (min_max[y_label, 1] - min_max[y_label, 0])
                all_pred_norm.extend(pred_value_normed.cpu().detach().numpy())
                all_y_norm.extend(y_norm.cpu().detach().numpy())
                all_y_label.extend(y_label.cpu().detach().numpy())
                loss = crierion(pred_value_normed.float(), y_norm)
                val_loss += loss.item()
            all_pred_norm = np.array(all_pred_norm)
            all_y_norm = np.array(all_y_norm)
            all_y_label = np.array(all_y_label)
            mae_per_label = []
            ranges = label_range[all_y_label]
            all_pred = all_pred_norm * ranges + min_max[all_y_label, 0].detach().cpu().numpy()
            all_y = all_y_norm * ranges + min_max[all_y_label, 0].detach().cpu().numpy()
            normed_mae_error = np.abs(all_pred_norm - all_y_norm).mean()
            normed_rmse_error = np.sqrt(np.square(all_pred_norm - all_y_norm).mean())
            mae_error = np.abs(all_pred - all_y)
            mse_error = np.square(all_pred - all_y)
            mertic_per_label = dict()
            for i in range(n_num_relation):
                mertic_per_label[i] = dict()
                mertic_per_label[i]['mae'] = mae_error[all_y_label == i].mean()
                mertic_per_label[i]['rmse'] = np.sqrt(mse_error[all_y_label == i].mean())
                mertic_per_label[i]['r2'] = r2_score(all_y[all_y_label == i], all_pred[all_y_label == i])

            val_mertic_per_label = mertic_per_label
            val_loss /= val_num_batches
            val_all_normed_mertic = {'mae': normed_mae_error, 'rmse': normed_rmse_error}
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                best_val_mertic_per_label = mertic_per_label
                all_embedding = {'entity_embedding': entity_embedding, 'relation_embedding': relation_embedding, 'num_relation_embedding': num_relation_embedding, 'reg_embedding': reg_embedding, 'missing_token_embedding': missing_token_embedding}
                torch.save(all_embedding,  embedding_save_path)
                torch.save(model.state_dict(), model_save_path)
            else:
                patience += 1
            if patience >= max_patience:
                break
        if epoch % 10 == 0:
            logging.info(f"Epoch {epoch}, train_loss: {train_loss}, val_loss: {val_loss}")

    logging.info(f"Best val loss: {best_val_loss}")
    logging.info(f"Best val mertic_per_label: {best_val_mertic_per_label}")



    if args.do_test:
        logging.info("Start testing")
        model.load_state_dict(torch.load(model_save_path))
        all_embedding = torch.load(embedding_save_path)
        entity_embedding = all_embedding['entity_embedding']
        relation_embedding = all_embedding['relation_embedding']
        num_relation_embedding = all_embedding['num_relation_embedding']
        reg_embedding = all_embedding['reg_embedding']
        missing_token_embedding = all_embedding['missing_token_embedding']
        test_loss = 0
        model.eval()
        all_y_norm = []
        all_pred_norm = []
        all_y_label = []
        with torch.no_grad():
            for batch_id in range(test_num_batches):
                start_id = batch_id * batch_size
                end_id = min((batch_id + 1) * batch_size, len(test_indices))
                index = test_indices[start_id:end_id]
                if args.topk:
                    y, y_label, embedding, path_counts, value, value_bit, reg_token_indices, path, value_label = hyperbolic_pruning(all_query, index, entity_embedding, relation_embedding, num_relation_embedding, reg_embedding, missing_token_embedding, one_hop_path, two_hop_path, three_hop_path, sample_num = sample_num, one_hop_sample_num= one_hop_sample_num,two_hop_sample_num=two_hop_sample_num ,three_hop_sample_num=three_hop_sample_num, topk=True)
                else:
                    y, y_label, embedding, path_counts, value, value_bit, reg_token_indices, path, value_label = hyperbolic_pruning(all_query, index, entity_embedding, relation_embedding, num_relation_embedding, reg_embedding, missing_token_embedding, one_hop_path, two_hop_path, three_hop_path, sample_num = sample_num, one_hop_sample_num= one_hop_sample_num,two_hop_sample_num=two_hop_sample_num ,three_hop_sample_num=three_hop_sample_num)
                value = torch.tensor(value, dtype=torch.float32).to(device)
                valu_bit = torch.tensor(value_bit, dtype=torch.float32).to(device)
                y = torch.tensor(y).float().to(device)
                y_label = torch.tensor(y_label).to(device)
                y_norm = (y - min_max[y_label, 0])/ (min_max[y_label, 1] - min_max[y_label, 0])
                value_ratio, weight = model(embedding, valu_bit, reg_token_indices.to(int).to(device), torch.tensor(path_counts).to(int).to(device))
                if args.add_model:
                    pred_value = value_ratio[:, 0] + value
                    pred_value = pred_value * value_ratio[:, 1]
                else:
                    pred_value = value_ratio.squeeze(1) * value
                pred_value = pred_value * weight.squeeze(1)
                pred_value = scatter(pred_value, torch.tensor(path_counts).to(int).to(device), reduce='sum')
                pred_value_normed = (pred_value - min_max[y_label, 0])/ (min_max[y_label, 1] - min_max[y_label, 0])
                all_pred_norm.extend(pred_value_normed.cpu().detach().numpy())
                all_y_norm.extend(y_norm.cpu().detach().numpy())
                all_y_label.extend(y_label.cpu().detach().numpy())
                loss = crierion(pred_value_normed.float(), y_norm)
                test_loss += loss.item()
            all_pred_norm = np.array(all_pred_norm)
            all_y_norm = np.array(all_y_norm)
            all_y_label = np.array(all_y_label)
            ranges = label_range[all_y_label]
            all_pred = all_pred_norm * ranges + min_max[all_y_label, 0].detach().cpu().numpy()
            all_y = all_y_norm * ranges + min_max[all_y_label, 0].detach().cpu().numpy()
            normed_mae_error = np.abs(all_pred_norm - all_y_norm).mean()
            normed_rmse_error = np.sqrt(np.square(all_pred_norm - all_y_norm).mean())
            test_all_normed_mertic = {'mae': normed_mae_error, 'rmse': normed_rmse_error}
            mae_error = np.abs(all_pred - all_y)
            mse_error = np.square(all_pred - all_y)
            mertic_per_label = dict()
            for i in range(n_num_relation):
                mertic_per_label[i] = dict()
                mertic_per_label[i]['mae'] = mae_error[all_y_label == i].mean()
                mertic_per_label[i]['rmse'] = np.sqrt(mse_error[all_y_label == i].mean())
                mertic_per_label[i]['r2'] = r2_score(all_y[all_y_label == i], all_pred[all_y_label == i])
            test_mertic_per_label = mertic_per_label
            test_loss /= test_num_batches
            logging.info(f"Test loss: {test_loss}")



if __name__ == "__main__":
    args = parse_args()
    main(args)
