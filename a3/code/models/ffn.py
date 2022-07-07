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

		return out

class FFNModel_Softmax(nn.Module):
	def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim):
		super(FFNModel_Softmax, self).__init__()
		
		self.layer_1 = nn.Linear(input_dim, hidden_dim_1) 
		self.actv_1 = nn.Softmax()
		self.layer_2 = nn.Linear(hidden_dim_1, hidden_dim_2)
		self.actv_2 = nn.Softmax()
		self.layer_3 = nn.Linear(hidden_dim_2, output_dim)  

	def forward(self, x):
		out = self.layer_1(x)
		out = self.actv_1(out)

		out = self.layer_2(out)
		out = self.actv_2(out)

		out = self.layer_3(out)

		return torch.softmax(out)

class FFNModel_Sigmoid(nn.Module):
	def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim):
		super(FFNModel_Sigmoid, self).__init__()
		
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