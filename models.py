from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
import random
from sklearn import metrics
from collections import OrderedDict
import timeit

class GCN(torch.nn.Module):
    def __init__(self, params, device):
        super().__init__()
        # self.pep_embedding = nn.Embedding(len(amino_to_idx), params["embedding_dim"])
        # self.pep_embedding.weight.data[:, :10] = torch.tensor(model_embedd)
        # self.pep_embedding.weight.requires_grad = True
        self.device = device
        self.conv1 = GCNConv(params["embedding_dim"], params["layer_size"] , normalize=True)  # num_node_features
        self.conv_out = GCNConv(params["layer_size"], params["encoding_dim"], normalize=True)  # num_classes
        self.dropout_rate = params["dropout"]
        self.fc_1 = nn.Linear(params["encoding_dim"], 1)


    def forward(self, data):
        data = data.to(self.device)
        x, edge_index = data.x, data.edge_index
        # x = self.pep_embedding(x)
        # x = torch.squeeze(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv_out(x, edge_index)

        x = self.fc_1(x)
        x = torch.sigmoid(x)

        return x  


        

class BiLSTM(nn.Module):
    def __init__(self, params, model_embedd, device):
        super().__init__()
        self.initialization = params["initialization"]
        self.device = device
        if params["initialization"] == "Random":
            self.pep_embedding = nn.Embedding(20, params["embedding_dim"])
        elif params["initialization"] in ["Kidera", "Kidera+bio"]:
            self.pep_embedding = nn.Embedding(20, params["embedding_dim"])
            self.pep_embedding.weight.data[:, :10] = torch.tensor(model_embedd)
            self.pep_embedding.weight.requires_grad = True

        padding = 8 if params["initialization"] == "Kidera+bio" else 0
        self.LSTM_cdr_encoder = nn.LSTM(
            params["embedding_dim"]+ padding,
            params["encoding_dim_lstm"],
            num_layers=params["lstm_num_layer"],
            batch_first=True,
            dropout=params["dropout"],
            bidirectional=True)

        self.fc = nn.Linear(params["encoding_dim_lstm"] * 2, params["encoding_dim_lstm"])

        self.fc_1 = nn.Linear(params["encoding_dim_lstm"], 1)

    def forward(self, dict_x):
        x = dict_x["embeding_pro"].to(self.device)

        if self.initialization in ["Kidera", "Kidera+bio", "Random"]:
            x = self.pep_embedding(x)
            if self.initialization == "Kidera+bio":
                aa_prop =  dict_x["aa_prop"].to(self.device)
                x = torch.cat((x, aa_prop), dim=2)

        self.LSTM_cdr_encoder.flatten_parameters()
        x, _ = self.LSTM_cdr_encoder(x)
        x = self.fc(x)

        x = self.fc_1(x)
        x = torch.sigmoid(x)

        return x



def predict_model(test_batches, kidera_embedding, params):
    device = params["device"]
    model = BiLSTM(params, kidera_embedding, device).to(device) if params["model"] == "BiLSTM" else  GCN(params, device).to(device)
    model_path = params["BiLSTM_model"] if params["model"] == "BiLSTM" else params["GCN_model"]
    start = timeit.default_timer()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    load_time = timeit.default_timer() -start
    print("load time", load_time)


    dict_res = evaluation(test_batches, model, 0, device, params, predict = True )

    return dict_res 



def loss_func(sings_real, sings_pred):
    sings_pred = torch.squeeze(sings_pred)
    loss = F.binary_cross_entropy(sings_pred, torch.squeeze(sings_real))
    return loss

def train_epoch(batches, model, optimizer, epoch, device, params):
    random.shuffle(batches)
    total_loss = 0
    total_signs_real, total_sign_pred = [], []
    data_size = 0
    model.train()
    for data in batches:
        optimizer.zero_grad()
        if params["model"] == "GCN":
            data, protein, name_pro = data[0], data[1], data[2]
        out = model(data)
        sign = data.to(device).y if params["model"] == "GCN" else data["signs"].to(device)
        if params["model"] == "BiLSTM":
            out = out.reshape((out.shape[0] * out.shape[1], out.shape[2]))
        loss = loss_func(sign, out)
 
        loss.backward(retain_graph=True)
        optimizer.step()

        batch_size = out.size(0)
        total_loss += (loss.item() * batch_size)
        data_size += batch_size

        sings_pred = torch.squeeze(out, 1)
        real = [float(item) for item in sign]
        total_signs_real += real
        pred = [float(item) for item in sings_pred]
        total_sign_pred += pred

    total_loss = total_loss / data_size
    fpr, tpr, _ = metrics.roc_curve(total_signs_real, total_sign_pred)
    auc = metrics.auc(fpr, tpr)
    print(f"epoch: {epoch} | train loss:{total_loss} | train auc:{auc}")
    return total_loss, auc


def evaluation(batches, model, epoch, device, params, write_res = True, predict =  False):
    f_predict = open(params["predict_file"], "w")
    total_loss = 0
    total_signs_real, total_sign_pred = [], []
    data_size = 0
    model.eval()
    dict_res = OrderedDict()
    for data in batches:
        if params["model"] == "GCN":
            data, protein, name_pro = data[0], data[1], data[2]
        out = model(data)
        sign = data.to(device).y if params["model"] == "GCN" else data["signs"].to(device)
        if params["model"] == "BiLSTM":
            out = out.reshape((out.shape[0] * out.shape[1], out.shape[2]))
            protein = data["protein"]
            name_pro = data["name_pro"]

        sings_pred = torch.squeeze(out, 1)
        pred = [float(item) for item in sings_pred]
        if not predict:
            loss = loss_func(sign, out)
            batch_size = out.size(0)
            total_loss += (loss.item() * batch_size)
            data_size += batch_size
            real = [float(item) for item in sign]
            total_signs_real += real
            total_sign_pred += pred


        for i, score in enumerate(pred):
            if predict:
                dict_res[f"{name_pro},{i + 1},{protein[i]}"] = score
            if write_res:
                f_predict.write(f"{name_pro},{i+1},{protein[i]},{score}\n")
    f_predict.close()
    if not predict:
        total_loss = total_loss / data_size
        fpr, tpr, _ = metrics.roc_curve(total_signs_real, total_sign_pred)
        auc = metrics.auc(fpr, tpr)
        print(f"epoch: {epoch} | val loss:{total_loss} | val auc:{auc}")
        return total_loss, auc
    else:
        return dict_res


def train_model2(train_batches, val_batches, test_batches, kidera_embedding,
                epochs, description, save_dir, params, device, early_stopping=True):

    model = BiLSTM(params, kidera_embedding, device).to(device) if params["model"] == "BiLSTM" else  GCN(params, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["w_d"])

    train_loss_list = list()
    val_loss_list = list()
    train_auc_list, val_auc_list, val_auc_per_pro_list, test_loss_list, test_auc_list = [], [], [], [], []
    max_val_auc = 0
    early_stopping_counter = 0
    best_model = 'None'
    for epoch in range(epochs):
        print(f'Epoch: {epoch + 1} / {epochs}')

        train_loss, train_auc = train_epoch(train_batches, model, optimizer, epoch +1, device, params)
        train_loss_list.append(train_loss)
        train_auc_list.append(train_auc)
        val_loss, val_auc = evaluation(val_batches, model, epoch + 1, device, params, False)  ##change to def eval
        val_loss_list.append(val_loss)
        val_auc_list.append(val_auc)
        test_loss, test_auc = evaluation(test_batches, model, epoch, device, params)  ##change to def eval 
        #test_loss_list.append(test_loss)
        #test_auc_list.append(test_auc)
        # val_auc_per_pro_list.append(val_auc_per_pro)

        if val_auc > max_val_auc:
            max_val_auc = val_auc
            early_stopping_counter = 0
            best_model = copy.deepcopy(model)
        elif early_stopping and early_stopping_counter == 10:
            break
        else:
            early_stopping_counter += 1


    #plot_loss(train_loss_list, val_loss_list, test_loss_list, num_epochs, save_dir, "loss", description)
    #plot_loss(train_auc_list, val_auc_list, test_auc_list, num_epochs, save_dir, "AUC", description)
    torch.save(best_model.state_dict(), f"{save_dir}/model_{description}.pt")
    #train_loss, train_auc = train_epoch(train_batches, best_model, optimizer, epoch, device)
    predict_file = params["predict_file"]
    predict_file = predict_file.split(".")
    params["predict_file"] = predict_file[0] + "_" + description + "_val" + predict_file[1]
    val_loss, val_auc = evaluation(val_batches, best_model, epoch, device, params, True)
    params["predict_file"] = predict_file[0] + "_" + description + "_test" + predict_file[1]
    test_loss, test_auc = evaluation(test_batches, best_model, epoch, device, params, True)  
    print(
        f"FINAL TEST best_model {description} : epoch {epoch + 1} | Test loss {test_loss} | Test auc {test_auc} "
        f"| val auc {val_auc}")

    return test_auc, train_auc, val_auc




