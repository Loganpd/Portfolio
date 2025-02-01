import time
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

"""
# Flow of actions
- 2 input sequences (event codes, tdelta) are fed into the model
    - the event codes sequence passes through an Embedding layer
        - this practically means that every event code is mapped into a vector of length = d_model
    - the tdelta sequence passes through a Linear layer. The tricky part is to understand how every tdelta entry is mapped into a vector of length = d_model (similar to the embedding layer outputs.) => # todo: how is this done?
    - the output dimensions of both of these layers are (sequence length)*d_model
- the input to the decoder stack for every singular input is of shape = 2*(sequence length)*d_model => # todo: how to feed this into the decoder stack?
- the output of the decoder stack is the extracted context.
    - the context is fed into a dense layer to predict the event code using a softmax activation function and sparse categorical cross entropy loss.
        - the difference between the output event code and its true event code is then back propagated.
    - the context AND the TRUE target label are fed into a dense layer to predict the value of tdelta
- the resulting event code output and tdelta are supposed to describe the next event that occurs in the input sequence.
"""

"""
# Notes
- max sequence length is set to 37. There are a total of 9 different event codes.
- Data normalization
    - All the tdeltas are normalized (divided) by the max observed tdelta=2591683. As a result, multiply the tdelta output of the network by 2591683 in order to get the generated tdelta in seconds (generation phase).
    - All the patient ages are divided ny the max observed age=91. Whatever age one would like to generate data for should be divided by this number before being fed into the network.
"""


##############################################################################################
################################### Dataset Class ############################################
class CustomDataset(Dataset):
    """Prepares the needed custom dataset class in pytorch."""

    def __init__(self, x, target_event, target_time_delta):
        """
        x: input data
        target_event: target event code
        target_time: target time delta
        """
        self.x = torch.Tensor(x)
        self.target_event = torch.Tensor(target_event)
        self.target_time_delta = torch.Tensor(target_time_delta)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x[idx], self.target_event[idx], self.target_time_delta[idx]


##############################################################################################
############################# Positional Encoding ############################################
def positional_encoding(sequence_length, d_model):
    """
    The output shape of the positional encoding should be (max_seq_len, d_model)
    """
    position = np.arange(sequence_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pos_encoding = np.zeros((sequence_length, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)

    return torch.tensor(pos_encoding, dtype=torch.float64).unsqueeze(0)


##############################################################################################
############################# Transformer components #########################################

class InputLayer(nn.Module):
    """Input Layer. The input layer in BF is fully connected and simply maps the input
        data with dimension d_input to a representation with dimension d_model, which is used
        throughout the decoder stack"""

    def __init__(self, device, input_length, d_model, sequence_length, dropout=0.1, activation=nn.GELU()):
        super(InputLayer, self).__init__()

        self.d_model = d_model
        self.sequence_length = sequence_length
        self.dense0 = nn.Linear(input_length, 512, device=device)
        self.dense = nn.Linear(512, d_model * sequence_length, device=device)
        # self.dropout0 = nn.Dropout(dropout)
        # self.dropout = nn.Dropout(dropout)
        # self.dense_activation0 = activation
        # self.dense_activation = activation

    def forward(self, x):
        x = self.dense0(x)
        # x = self.dropout0(x)
        # x = self.dense_activation0(x)

        x = self.dense(x)
        # x = self.dropout(x)
        # x = self.dense_activation(x)
        return x.view(x.shape[0], self.sequence_length, self.d_model)


class DecoderLayer(nn.Module):
    """Masked multi-headed attention (aka MMHA) layer + dense layer combo"""

    def __init__(self, device, d_model, sequence_length, num_heads, dropout=0.1, activation=nn.GELU()):
        super(DecoderLayer, self).__init__()
        # self.attn_mask = torch.triu(torch.ones(sequence_length, sequence_length) * float('-inf'), diagonal=1).to(device)
        # self.attn_mask = torch.tril(torch.ones(sequence_length, sequence_length)).to(device)
        self.attn_mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).bool().to(device)

        self.device = device
        self.d_model = d_model
        self.sequence_length = sequence_length

        # MMHA chunk
        self.attention_layer = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout,
                                                     device=device, batch_first=True)
        self.dense = nn.Linear(in_features=d_model * sequence_length, out_features=d_model * sequence_length,
                               device=device)
        self.dropout = nn.Dropout(dropout)
        # self.activation = activation

        # layer normalization
        self.layer_norm0 = nn.LayerNorm(d_model, device=device)
        self.layer_norm = nn.LayerNorm(d_model, device=device)

    def forward(self, x) -> Tensor:
        skip = x

        # masked multi-headed attention - x.shape = (batch size, sequence length, d_model)
        x, _ = self.attention_layer(x, x, x, attn_mask=self.attn_mask)
        x += skip
        x = self.layer_norm0(x)
        skip = x
        x = self.dense(x.reshape(x.shape[0], -1)).reshape(x.shape[0], self.sequence_length, self.d_model)
        x = self.dropout(x)
        # x = self.activation(x)

        # skip connection addition and layer norm
        x += skip
        x = self.layer_norm(x)
        return x


class ContextExtraction(nn.Module):
    """This class extracts the context from the input sequence through a stack of decoder layers."""

    def __init__(self, device, decoder_stack_count=4, d_model=64, sequence_length=50, num_heads=8, dropout=0.1,
                 activation=nn.GELU()):
        """
        param: embedding_dim: dimension of embedding layer also known as d_model
        """
        super(ContextExtraction, self).__init__()
        self.layers = nn.ModuleList()
        for layer in range(decoder_stack_count):
            self.layers.append(
                DecoderLayer(device=device, d_model=d_model, sequence_length=sequence_length, num_heads=num_heads,
                             dropout=dropout,
                             activation=activation))

    def forward(self, x):
        for decoder_layer in self.layers:
            x = decoder_layer(x)
        return x


class OutputLayers(nn.Module):
    """The model has 2 output layers:
        1. event code prediction
        2. tdelta prediction
        The prediction of the event code is used for tdelta prediction"""

    def __init__(self, device, d_model, sequence_length, ecode_count=9):
        super(OutputLayers, self).__init__()
        # event code output
        self.ecode_dense = nn.Linear(in_features=d_model * sequence_length, out_features=ecode_count, device=device)
        # self.ecode_activation = nn.Softmax(dim=-1)  # not needed in pytorch when using nn.CrossEntropyLoss() or the
        # nn.functional.cross_entropy() version of it.

        # tdelta output
        self.tdelta_dense = nn.Linear(in_features=d_model * sequence_length + ecode_count, out_features=1,
                                      device=device)
        # self.relu = nn.ReLU()

    def forward(self, context, event_code, train=True):
        """The event_code must be one hot encoded"""
        ecode = self.ecode_dense(context)
        # ecode = self.ecode_activation(ecode)  # not needed since loss function does it itself

        if train:
            tdelta = self.tdelta_dense(torch.cat((context, event_code), 1))
            # tdelta = self.relu(tdelta)
            return ecode, tdelta
        else:
            vector = ecode.new_zeros(ecode.shape)
            vector[0, ecode.argmax()] = 1
            tdelta = self.tdelta_dense(torch.cat((context, vector), 1))
            # tdelta = self.relu(tdelta)
            return vector, tdelta


def _mse_loss(prediction, target):
    return F.mse_loss(prediction.squeeze(), target) * 5


def _cross_entropy_loss(prediction, target):
    """
    Target should be the one-hot encoded version of the target labels.
    Prediction should be a dense layer with as many units as there are classes, without any activation function!!!
    """
    return F.cross_entropy(prediction.squeeze(), target)


class Healsformer:
    def __init__(self, device, network, epochs, learning_rate, optimizer='adam'):
        self.device = device
        self.network = network
        self.epochs = epochs
        self.learning_rate = learning_rate
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate, amsgrad=False)
        elif optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.learning_rate, nesterov=True,
                                             momentum=0.9)
        else:  # amsgrad
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.25,
                                                           total_iters=epochs * 2 // 3)

        self.train_loader = None
        self.test_loader = None
        self.val_loader = None

        self.best_val_loss = float('inf')
        self.best_model_state_dict = None
        self.best_model_path = "BestModelStateDict.pt"

        self.training_total_loss = []
        self.validation_total_loss = []

        self.event_code_decoder = {0: 'ED_intime',
                                   1: 'ED_outtime',
                                   2: 'CT_time',
                                   3: 'SW_time',
                                   4: 'TPA_time',
                                   5: 'admission_time',
                                   6: 'discharge_time',
                                   7: 'icuin',
                                   8: 'icuout'}

    def load_data(self, x, y_events, y_tdelta, test_split=0.15, val_split=0.15, batch_size=32):
        # test split is applied on all the data. val split is applied onto the training data.
        x_train, x_test, y0_train, y0_test, y1_train, y1_test = train_test_split(x,
                                                                                 y_events,
                                                                                 y_tdelta,
                                                                                 test_size=test_split,
                                                                                 random_state=42)
        x_train, x_val, y0_train, y0_val, y1_train, y1_val = train_test_split(x_train,
                                                                              y0_train,
                                                                              y1_train,
                                                                              test_size=val_split,
                                                                              random_state=42)

        train_data = CustomDataset(x_train, y0_train, y1_train)
        test_data = CustomDataset(x_test, y0_test, y1_test)
        val_data = CustomDataset(x_val, y0_val, y1_val)

        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    def _train(self):
        self.network.train()
        for x, event_true, tdelta_true in self.train_loader:
            self.optimizer.zero_grad()
            event_pred, tdelta_pred = self.network(x.to(self.device), event_true.to(self.device))
            event_loss = _cross_entropy_loss(event_pred, event_true.to(self.device))
            tdelta_loss = _mse_loss(tdelta_pred, tdelta_true.to(self.device))
            total_loss = event_loss + tdelta_loss
            total_loss.backward()
            self.optimizer.step()
        return total_loss

    def _validate(self):
        self.network.eval()
        total_losses = []
        event_losses = []
        tdelta_losses = []
        for x, event_true, tdelta_true in self.val_loader:
            event_pred, tdelta_pred = self.network(x.to(self.device), event_true.to(self.device))
            event_loss = _cross_entropy_loss(event_pred, event_true.to(self.device))
            tdelta_loss = _mse_loss(tdelta_pred, tdelta_true.to(self.device))
            event_losses.append(event_loss.cpu().detach())
            tdelta_losses.append(tdelta_loss.cpu().detach())
            total_loss = event_loss + tdelta_loss
            total_losses.append(total_loss.cpu().detach())
        val_loss = np.array(total_losses).mean()
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model_state_dict = self.network.state_dict()
            torch.save(self.best_model_state_dict, self.best_model_path)
        return val_loss, np.array(event_losses).mean(), np.array(tdelta_losses).mean()

    def test(self):
        self.network.eval()
        total_losses = []
        for x, event_true, tdelta_true in self.test_loader:
            event_pred, tdelta_pred = self.network(x.to(self.device), event_true.to(self.device))
            event_loss = _cross_entropy_loss(event_pred, event_true.to(self.device))
            tdelta_loss = _mse_loss(tdelta_pred, tdelta_true.to(self.device))
            total_loss = event_loss + tdelta_loss
            total_losses.append(total_loss.cpu().detach())
        mean_event_loss = np.array(total_losses).mean()
        print(f"Test set performance: Mean total loss = {mean_event_loss:.4f}")

    def train(self, verbose=True):
        start = time.time()
        for i in range(self.epochs):
            epoch_start = time.time()
            train_total_loss = self._train()
            self.training_total_loss.append(train_total_loss.cpu().detach())

            val_total_loss, val_event_loss, val_tdelta_loss = self._validate()
            self.validation_total_loss.append(val_total_loss)

            epoch_end = time.time()
            if verbose:
                print(
                    f"Epoch {i + 1:<3}--> Training: total loss={train_total_loss:.4f} -- Validation: total loss={val_total_loss:.4f} (Event loss={val_event_loss:.4f} + Tdelta loss={val_tdelta_loss:.4f}) -- Run time: {epoch_end - epoch_start:.2f} seconds.")
            self.scheduler.step()
        end = time.time()
        print(f"Training for {self.epochs} epochs completed in {end - start:.2f} seconds.")
        print(f"Best validation loss={self.best_val_loss}")
        self.network.load_state_dict(torch.load(self.best_model_path))

    def plot_error_vs_epoch(self):
        fig = plt.figure()
        plt.grid(zorder=0)
        plt.xlabel("Epochs")
        plt.ylabel("Total error")
        plt.title("Error vs Epoch")
        plt.plot(self.training_total_loss, label="Training", color="blue")
        plt.plot(self.validation_total_loss, label="Validation", color="orange")
        plt.ylim(bottom=0)
        # plt.xlim(left=-0.1)
        plt.legend()
        plt.show()

    def generate(self, max_sequence_length=37):
        def get_triplet():
            sex = np.random.choice(['M', 'F'], p=[0.526, 0.474])
            age_bins = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
            age_bin_probs = [0.00344149, 0.00516224, 0.00811209, 0.00999672, 0.01655195,
                             0.03105539, 0.0483448, 0.07276303, 0.10660439, 0.13430023,
                             0.13413635, 0.13798755, 0.13356277, 0.0950508, 0.06293019]
            age = np.random.choice(age_bins, p=age_bin_probs)
            age = np.random.choice(range(age, age+5))
            first_event = np.random.choice([0, 5], p=[0.725, 0.275])
            return age, sex, first_event

        age, sex, first_event = get_triplet()
        age = [age/91]
        sex = [0, 1] if sex == 'M' else [1, 0]
        ed_intime = [1] + [0] * 8                # of length 9 (onehot encoded event)
        admission_time = [0] * 5 + [1] +[0]*3    # of length 9 (onehot encoded event)
        starting_event = ed_intime if first_event == 0 else admission_time

        end_of_sequence = [0] * 6 + [1] + [0] * 2  # of length 9 (onehot encoded event) - eos is discharge time
        tdeltas = [1e-12 / 2591683] + (max_sequence_length-1) * [0]
        sequence = age + sex + starting_event
        input_vector = torch.tensor(sequence + (max_sequence_length-1) * end_of_sequence + tdeltas, dtype=torch.float32).unsqueeze(0).to(self.device)

        ecode, tdelta = self.network(input_vector, None, train=False)
        sequence += ecode.cpu().tolist()[0]
        generated_event_count = 1
        tdeltas[generated_event_count] = tdelta.cpu().tolist()[0][0]

        while ecode[0, 6] != 1:  # while the end of sequence (discharge time) is not generated
            input_vector = torch.tensor(sequence + ((max_sequence_length-1) - generated_event_count) * end_of_sequence + tdeltas, dtype=torch.float32).unsqueeze(
                0).to(self.device)
            if generated_event_count != max_sequence_length-2:
                ecode, tdelta = self.network(input_vector, None, train=False)
                sequence += ecode.cpu().tolist()[0]
                generated_event_count += 1
                tdeltas[generated_event_count] = tdelta.cpu().tolist()[0][0]
            else:
                ecode, tdelta = self.network(input_vector, None, train=False)
                sequence += [0] * 6 + [1] + [0] * 2  # discharge time code
                tdeltas[-1] = tdelta.cpu().tolist()[0][0]
                break
        events = []
        sequence = sequence[3:]
        for i in range(len(sequence) // 9):
            events.append(self.event_code_decoder[np.argmax(sequence[9 * i:9 * (i + 1)])])

        return events, np.array(tdeltas) * 600000  # the tdeltas are returned in seconds

    def generate_n(self, n):
        generated_events = []
        generated_tdeltas = []
        for i in range(n):
            events, tdeltas = self.generate()
            generated_events.append(events)
            generated_tdeltas.append(tdeltas)
        return generated_events, generated_tdeltas


##############################################################################################
################################## Network definition ########################################
class Net(nn.Module):
    def __init__(self, device, input_length, d_model, sequence_length, decoders_in_stack, num_heads, dropout=0.1,
                 activation=nn.GELU()):
        super(Net, self).__init__()
        # there are 9 possible event codes.

        ### input layer ###
        self.input_layer = InputLayer(device, input_length=input_length, d_model=d_model,
                                      sequence_length=sequence_length, dropout=dropout,
                                      activation=activation)

        ### Positional encoding ###
        self.positional_encodings = positional_encoding(sequence_length=sequence_length, d_model=d_model).to(device)
        # Input embeddings shape: (batch_size, sequence_length, d_model)
        # Positional encoding shape: (sequence_length, d_model)

        ### decoder stack ###
        self.decoder_stack = ContextExtraction(device, decoder_stack_count=decoders_in_stack, d_model=d_model,
                                               sequence_length=sequence_length,
                                               num_heads=num_heads, dropout=dropout, activation=activation)

        ### output layers ###
        self.output_layers = OutputLayers(device, d_model=d_model, sequence_length=sequence_length)

    def forward(self, x, event_code, train=True):
        x = self.input_layer(x)
        x += self.positional_encodings
        x = self.decoder_stack(x)

        if train:
            event_code, tdelta = self.output_layers(x.view(x.shape[0], -1), event_code)
        else:
            event_code, tdelta = self.output_layers(x.view(x.shape[0], -1), None, train=False)
        return event_code, tdelta


if __name__ == "__main__":
    import pandas as pd
    from analysis import *

    print("Cuda is available:", torch.cuda.is_available())
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device('cuda')

    x, y_events, y_tdelta = joblib.load("PreparedData/x_eventy_tdeltay_8+.pkl")

    net1 = Net(device=device, input_length=x.shape[1], d_model=8, sequence_length=16, decoders_in_stack=1, num_heads=4,
               dropout=0.1, activation=nn.GELU())
    model = Healsformer(device=device, network=net1, epochs=2, learning_rate=0.001)
    model.load_data(x=x, y_events=y_events, y_tdelta=y_tdelta, test_split=0.15, val_split=0.15, batch_size=16)
    model.train()
    model.test()
    model.plot_error_vs_epoch()

    event_sequences, tdelta_sequences = model.generate_n(10)

    event_occurrence(event_sequences)
    stay_duration(event_sequences, tdelta_sequences)
    plot_histogram(event_sequences, tdelta_sequences, 'emergency')
    plot_histogram(event_sequences, tdelta_sequences, 'hospital')
    plot_histogram(event_sequences, tdelta_sequences, 'icu')
