import torch
import torch.nn as nn 
import torch.nn.functional as F

hidden_sizes=[64, 128, 256, 128, 64]
class MLP(nn.Module):
    def __init__(self, n_features=13, n_classes=13):
        super(MLP, self).__init__()
        # Match exact architecture of saved model
        self.fc1 = nn.Linear(n_features, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128) 
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.output = nn.Linear(64, n_classes)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.bn1(F.gelu(self.fc1(x)))
        x = self.dropout(x)
        x = self.bn2(F.leaky_relu(self.fc2(x)))
        x = self.dropout(x)
        x = self.bn3(F.elu(self.fc3(x)))
        x = self.dropout(x)
        return F.softmax(self.output(x), dim=1)

# Remove EnhancedMLP since we merged it into MLP
# Initialize model with correct architecture
class EnhancedNN(nn.Module):
    def __init__(self, n_features=13, n_classes=13, hidden_sizes=[64, 128, 256, 128, 64]):
        super(EnhancedNN, self).__init__()
        self.bn1 = nn.BatchNorm1d(n_features)
        self.fc1 = nn.Linear(n_features, hidden_sizes[0])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn4 = nn.BatchNorm1d(hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.bn5 = nn.BatchNorm1d(hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], hidden_sizes[4])
        self.bn6 = nn.BatchNorm1d(hidden_sizes[4])
        self.output = nn.Linear(hidden_sizes[4], n_classes)
        
        self.dropout = nn.Dropout(0.3)
        self.gelu = nn.GELU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.elu = nn.ELU()
        
    def forward(self, x):
        x = self.bn1(x)
        x = self.gelu(self.fc1(x))
        x = self.dropout(self.bn2(x))
        
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(self.bn3(x))
        
        x = self.elu(self.fc3(x))
        x = self.dropout(self.bn4(x))
        
        x = self.gelu(self.fc4(x))
        x = self.dropout(self.bn5(x))
        
        x = self.leaky_relu(self.fc5(x))
        x = self.dropout(self.bn6(x))
        
        return F.softmax(self.output(x), dim=1)



model = EnhancedNN()