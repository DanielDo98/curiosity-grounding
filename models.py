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
                300 + 1,  #This is because we train with a higher max episode length
                #args.max_episode_length+1,
                self.time_emb_dim)

        # Action embedding
        self.action_emb_dim = 8
        self.action_emb_layer = nn.Embedding(3, self.action_emb_dim) #Map from action to embedding

        self.frame_size = 64 * 8 * 17
        self.cnn_out_size = 5 * self.frame_size
        self.rep_size = 2 * self.cnn_out_size + self.gru_hidden_size
        # A3C-LSTM layers
        self.linear = nn.Linear(self.rep_size, 384)
        self.lstm = nn.LSTMCell(384, 256)
        self.critic_linear = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 + self.time_emb_dim, 3)

        #Inverse dynamic model
        self.inverse_linear = nn.Linear(2 * self.cnn_out_size, 256)
        self.inverse_actor = nn.Linear(256, 3)

        #Forward dynamic model. Action is embedded in 8-dim vector. Model predicts next final FRAME, not next 5 frames
        self.forward_linear = nn.Linear(self.cnn_out_size + self.action_emb_dim, 256)
        self.forward_state = nn.Linear(256, self.frame_size)

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
        x_attention = x_attention.expand(5, 64, 8, 17)
        print(x_attention.shape)
        print(x_image_rep.size())
        assert x_image_rep.size() == x_attention.size()
        x = x_image_rep*x_attention
        #Concatentation as suggested by Manning
        x = torch.cat((x.view(-1), x_image_rep.view(-1), x_instr_rep.view(-1))).unsqueeze(0)

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
        
        return x #this is logits over action. Embedding is just to process action.

    '''
    Predicts s_{t+1} given s_t and a_t. 0 < a_t < #actions
    Both s_t and a_t are variables
    #actions is 3 for now
    '''
    def forwardDynamicsForward(self, inputs):
        s1, a1 = inputs

        # Get the image representation for s1
        s1 = self.relu(self.conv1(s1))
        s1 = self.relu(self.conv2(s1))
        s1_image_rep = self.relu(self.conv3(s1)).view(-1)

        a1 = self.action_emb_layer(a1).view(-1)
        x = torch.cat((s1_image_rep, a1))

        x = self.forward_linear(x)
        x = self.forward_state(x)
        
        return x

    def getImageRep(self, s):
        s = self.relu(self.conv1(s))
        s = self.relu(self.conv2(s))
        s_image_rep = self.relu(self.conv3(s)).view(-1)
        return s_image_rep
    
    #Similar to teacherForward, but sets attention and instruction representation to 0.
    def curiousForward(self, inputs):
        x, _, (tx, hx, cx) = inputs

        # Get the image representation
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x_image_rep = self.relu(self.conv3(x))

        # Attention and instruction representation start out as 0
        x_instr_rep = torch.zeros((self.gru_hidden_size))
        x = torch.zeros((self.cnn_out_size))
        x = torch.cat((x.view(-1), x_image_rep.view(-1), x_instr_rep.view(-1))).unsqueeze(0)

        # A3C-LSTM
        x = self.relu(self.linear(x))
        hx, cx = self.lstm(x, (hx, cx))
        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
        

    def forward(self, inputs, teacher=True, inverse=True, curious=True):
        if teacher:
            return self.teacherForward(inputs)
        elif inverse:
            return self.inverseDynamicsForward(inputs)
        elif not curious:
            return self.forwardDynamicsForward(inputs)
        else:
            return self.curiousForward(inputs)
