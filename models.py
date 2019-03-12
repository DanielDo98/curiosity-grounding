import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class A3C_LSTM_GA(torch.nn.Module):

    def __init__(self, args):
        super(A3C_LSTM_GA, self).__init__()

        # General
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Image Processing
        self.conv1 = nn.Conv2d(3, 128, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2)

        # Instruction Processing
        self.gru_hidden_size = 256
        self.input_size = args.input_size
        self.embedding = nn.Embedding(self.input_size, 32)
        self.gru = nn.GRU(32, self.gru_hidden_size)

        # Gated-Attention layers
        self.attn_linear = nn.Linear(self.gru_hidden_size, 64)

        # Time embedding layer, helps in stabilizing value prediction.
        # max episode length is 30 by default.
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(
                args.max_episode_length+1,
                self.time_emb_dim)

        rep_size = 64 * 8 * 17
        # A3C-LSTM layers
        self.linear = nn.Linear(rep_size, 256)
        self.lstm = nn.LSTMCell(256, 256)
        self.critic_linear = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 + self.time_emb_dim, 3)

        #Inverse dynamic model
        self.inverse_linear = nn.Linear(2 * rep_size, 256)
        self.inverse_actor = nn.Linear(256, 3) #3 different actions for now TODO

        #Forward dynamic model. action is 1-hot
        self.forward_linear = nn.Linear(rep_size + 3, 256)
        self.forward_state = nn.Linear(256, rep_size)

        self.train()

    #Produces action for state s_t
    def teacherForward(self, inputs):
        x, input_inst, (tx, hx, cx) = inputs

        # Get the image representation
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x_image_rep = self.relu(self.conv3(x))

        # Get the instruction representation
        embedded = self.embedding(input_inst).permute(1,0,2)
        encoder_hidden = Variable(self.gru(embedded)[0][-1]) #Get last hidden state
        x_instr_rep = encoder_hidden

        # Get the attention vector from the instruction representation
        x_attention = self.sigmoid(self.attn_linear(x_instr_rep))

        # Gated-Attention
        x_attention = x_attention.unsqueeze(2).unsqueeze(3)
        x_attention = x_attention.expand(1, 64, 8, 17)
        assert x_image_rep.size() == x_attention.size()
        x = x_image_rep*x_attention
        x = x.view(x.size(0), -1)

        # A3C-LSTM
        x = self.relu(self.linear(x))
        hx, cx = self.lstm(x, (hx, cx))
        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)

    '''
    Predicts a_t based on s_t and s_{t+1}
    '''
    def inverseDynamicsForward(self, inputs):
        s1, s2 = inputs

        # Get the image representation for s1
        s1 = self.relu(self.conv1(s1))
        s1 = self.relu(self.conv2(s1))
        s1_image_rep = self.relu(self.conv3(s1)).view(-1)

        # Get the image representation for s2
        s2 = self.relu(self.conv1(s2))
        s2 = self.relu(self.conv2(s2))
        s2_image_rep = self.relu(self.conv3(s2)).view(-1)

        x = torch.cat((s1_image_rep, s2_image_rep))
        x = self.inverse_linear(x)
        x = self.inverse_actor(x)
        
        return x #this is logits for action

    '''
    Predicts s_{t+1} given s_t and a_t. a_t is a tensor of (#actions,)
    Both s_t and a_t are variables
    #actions is 3 for now
    '''
    def forwardDynamicsForward(self, inputs):
        s1, a1 = inputs

        # Get the image representation for s1
        s1 = self.relu(self.conv1(s1))
        s1 = self.relu(self.conv2(s1))
        s1_image_rep = self.relu(self.conv3(s1)).view(-1)

        a1 = a1.view(-1) 
        x = torch.cat((s1_image_rep, a1))
        x = self.forward_linear(x)
        x = self.forward_state(x)
        
        return x

    def getImageRep(self, s):
        s = self.relu(self.conv1(s))
        s = self.relu(self.conv2(s))
        s_image_rep = self.relu(self.conv3(s)).view(-1)
        return s_image_rep

    def forward(self, inputs, teacher=True, inverse=True):
        if teacher:
            return self.teacherForward(inputs)
        elif inverse:
            return self.inverseDynamicsForward(inputs)
        else:
            return self.forwardDynamicsForward(inputs)
