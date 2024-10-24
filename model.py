import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
from pathlib import Path

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.model = nn.Sequential (
            nn.Linear(input_size, hidden_size),
            # nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(hidden_size, output_size)
            # nn.ReLU()
        )

        self.def_model_name = 'model.pth'
        self.model_folder_path = Path('./model')
        self.score_path = 'plot_score.pickle'
        self.mean_scores_path = 'plot_mean_score.pickle'

    def forward(self, x):
        return(self.model(x))    

    def save(self, plot_scores, plot_mean_scores, file_name=None) -> None:
        if not file_name:
            file_name = self.def_model_name
        if not self.model_folder_path.exists():
            self.model_folder_path.mkdir()

        file_name = self.model_folder_path / Path(file_name)

        torch.save(self.state_dict(), str(file_name))
        pickle.dump(plot_scores, Path(self.model_folder_path, self.score_path).open('wb'))
        pickle.dump(plot_mean_scores, Path(self.model_folder_path, self.mean_scores_path).open('wb'))

        # print("Saving Model state as: ", str(file_name))

    def load (self, file_name=None) -> list:
        if file_name is None:
            file_name = self.def_model_name

        file_name = self.model_folder_path / file_name

        if file_name.exists():
            self.load_state_dict(torch.load(str(file_name)))
            # print("Loading Model state from: ", str(file_name))
            plot_scores = pickle.load( Path(self.model_folder_path, self.score_path).open('rb') )
            plot_mean_scores = pickle.load( Path(self.model_folder_path, self.mean_scores_path).open('rb') )

            return plot_scores, plot_mean_scores
        
        # print("Saved Model doesnt exists!")
        return [], []

    


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()


    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float16)
        reward = torch.tensor(reward, dtype=torch.float16)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                pred_nxt = self.model(next_state[idx])
                Q_new = reward[idx] + self.gamma * torch.max(pred_nxt)

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()



    # def train_step(self, state, action, reward, next_state):
    #     state = torch.tensor(state, dtype=torch.float)
    #     next_state = torch.tensor(next_state, dtype=torch.float)
    #     action = torch.tensor(action, dtype=torch.long)
    #     reward = torch.tensor(reward, dtype=torch.float)
    #     # (n, x)

    #     if len(state.shape) == 1:
    #         # (1, x)
    #         state = torch.unsqueeze(state, 0)
    #         next_state = torch.unsqueeze(next_state, 0)
    #         action = torch.unsqueeze(action, 0)
    #         reward = torch.unsqueeze(reward, 0)

    #     # 1: predicted Q values with current state
    #     pred = self.model(state)

    #     target = pred.clone()
    #     for idx in range(len(reward)):
    #         Q_new = reward[idx]
    #         Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

    #         target[idx][action[idx]] = Q_new
    
    #     # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
    #     # pred.clone()
    #     # preds[argmax(action)] = Q_new
    #     self.optimizer.zero_grad()
    #     loss = self.criterion(target, pred)
    #     loss.backward()

    #     self.optimizer.step()