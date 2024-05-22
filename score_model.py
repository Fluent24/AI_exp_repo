import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder,  TransformerEncoderLayer

class MLP(nn.Module):
    def __init__(self,input_dim, hidden_dim):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim) # hidden layer
        self.layer2 = torch.nn.Linear(hidden_dim, 1) # output layer
        self.relu = torch.nn.ReLU() # activation function

    def forward(self, x):
        out = self.layer2(self.relu(self.layer1(x)))
        return out
class MLP_multi_layers(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(MLP_multi_layers, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
class CNN_LSTM_RegressionModel(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=128, num_layers=2, dropout=0.1):
        super(CNN_LSTM_RegressionModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        # Adjust the LSTM input dimension based on the output of CNN
        cnn_output_dim = 64 * (input_dim // 16)
        self.lstm = nn.LSTM(cnn_output_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size, feature_dim = x.size()
        x = x.view(batch_size, 1, 32, -1)  # Adjust to [batch_size, 1, height, width] format
        x = self.cnn(x)
        x = x.view(batch_size, -1)
        x = x.unsqueeze(1)  # [batch_size, 1, cnn_output_dim]
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Get the last output of the LSTM
        x = self.fc(x)
        return x
    
class TransformerRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, output_dim, dropout=0.1):
        super(TransformerRegressionModel, self).__init__()
        
        # 입력 데이터를 TransformerEncoder에 적합한 형태로 변환하는 임베딩 레이어
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # TransformerEncoderLayer를 여러 개 쌓아 TransformerEncoder 구성
        encoder_layer = TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 드롭아웃 레이어
        self.dropout = nn.Dropout(dropout)
        
        # 최종 출력을 위한 선형 레이어
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 입력 데이터를 임베딩하고 위치 정보 추가
        x = self.embedding(x)
        x = x.unsqueeze(1)  # [batch_size, seq_len, hidden_dim] 형태로 변환
        
        # TransformerEncoder로 입력 데이터 처리
        x = self.transformer_encoder(x)
        
        # 드롭아웃 적용
        x = self.dropout(x)
        
        # 최종 출력 값 계산 (평균 풀링 후 선형 변환)
        x = x.mean(dim=1)  # 각 시퀀스의 평균 값 계산
        x = self.fc(x)
        return x