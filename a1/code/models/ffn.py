import torch.nn as nn
import torch

class FFNModel(nn.Module):
	def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim):
		super(FFNModel, self).__init__()
		
		self.layer_1 = nn.Linear(input_dim, hidden_dim_1) 
		self.actv_1 = nn.Tanh()
		self.layer_2 = nn.Linear(hidden_dim_1, hidden_dim_2)
		self.actv_2 = nn.Tanh()
		self.layer_3 = nn.Linear(hidden_dim_2, output_dim)  

	def forward(self, x):
		out = self.layer_1(x)
		out = self.actv_1(out)

		out = self.layer_2(out)
		out = self.actv_2(out)

		out = self.layer_3(out)

		return torch.sigmoid(out)