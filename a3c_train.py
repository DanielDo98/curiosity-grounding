import torch.optim as optim
import env as grounding_env

from models import *
from torch.autograd import Variable
import torch.nn.functional as F

import logging

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(rank, args, shared_model):
    torch.manual_seed(args.seed + rank)

    env = grounding_env.GroundingEnv(args)
    env.game_init()

    model = A3C_LSTM_GA(args)

    if (args.load != "0"):
        print(str(rank) + " Loading model ... "+args.load)
        model.load_state_dict(
            torch.load(args.load, map_location=lambda storage, loc: storage))

    model.train()

    optimizer = optim.SGD(shared_model.parameters(), lr=args.lr)

    p_losses = []
    v_losses = []

    (image, instruction), _, _, _ = env.reset()
    instruction_idx = []
    for word in instruction.split(" "):
        instruction_idx.append(env.word_to_idx[word])
    instruction_idx = np.array(instruction_idx)

    image = torch.from_numpy(image).float()/255.0
    instruction_idx = torch.from_numpy(instruction_idx).view(1, -1)

    done = True

    #Curiosity bookkeeping
    prevState = image
    prevAction = None
    beta = .2
    lamb = .2 #TODO tune this. Language grounding is important
    eta = 1 #TODO tune this hyperparameter


    episode_length = 0
    num_iters = 0
    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            episode_length = 0
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))

        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)

        values = []
        log_probs = []
        rewards = []
        entropies = []

        #Optimizing over this
        policy_loss = Variable(torch.zeros(1,1))
        value_loss = 0

        for step in range(args.num_steps):
            episode_length += 1
            tx = Variable(torch.from_numpy(np.array([episode_length])).long())

            value, logit, (hx, cx) = model((Variable(image.unsqueeze(0)),
                                            Variable(instruction_idx),
                                            (tx, hx, cx)), teacher=True, inverse=False)
            prob = F.softmax(logit, dim=1)
            log_prob = F.log_softmax(logit, dim=1)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy) 

            action = prob.multinomial(1).data #action is sampled once from multinomial
            log_prob = log_prob.gather(1, Variable(action))
            oldAction = action

            action = action.numpy()[0, 0]
            curReward = 0
            (image, _), reward, done,  _ = env.step(action)
            curReward += reward
           

            done = done or episode_length >= args.max_episode_length

            if done:
                (image, instruction), _, _, _ = env.reset()
                instruction_idx = []
                for word in instruction.split(" "):
                    instruction_idx.append(env.word_to_idx[word])
                instruction_idx = np.array(instruction_idx)
                instruction_idx = torch.from_numpy(
                        instruction_idx).view(1, -1)

            image = torch.from_numpy(image).float()/255.0

            curReward = 0
            #curiosity loss and reward
            if prevAction is not None:
                pred_action = model((Variable(prevState.unsqueeze(0)),
                                Variable(image.unsqueeze(0))), 
                                teacher=False, inverse=True)
                a_prob = F.softmax(pred_action) 
                a_loss = 1/2 * torch.norm(a_prob - prob)
                #Because we have access to softmax, might as well use it TODO
                actionTensor = torch.eye(3)[prevAction[0]]

                pred_state = model((Variable(prevState.unsqueeze(0)),
                                Variable(actionTensor)), teacher=False, inverse=False)

                s_loss = 1/2 * torch.norm(pred_state - model.getImageRep(Variable(image.unsqueeze(0))))
                policy_loss += (1-beta) * a_loss + beta * s_loss
                curReward += eta * s_loss.item()

            #Updating curiosity
            prevAction = oldAction
            prevState = image

            values.append(value) #critic in actor-critic
            log_probs.append(log_prob)
            rewards.append(curReward) #+2 if found, -.1 if not found, plus intrinsic

            if done:
                break
        
        R = torch.zeros(1,1)
        if not done:
            tx = Variable(torch.from_numpy(np.array([episode_length])).long())
            value, _, _ = model((Variable(image.unsqueeze(0)),
                                 Variable(instruction_idx), (tx, hx, cx)),
                                 teacher=True, inverse=False)
            R = value.data

        values.append(Variable(R))
        R = Variable(R)

        new_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            new_loss = new_loss - \
                log_probs[i] * Variable(gae) - 0.01 * entropies[i]

        policy_loss += lamb * new_loss
        optimizer.zero_grad()

        p_losses.append(policy_loss.data[0, 0])
        v_losses.append(value_loss.data[0, 0])

        if(len(p_losses) > 1000):
            num_iters += 1
            print(" ".join([
                  "Training thread: {}".format(rank),
                  "Num iters: {}K".format(num_iters),
                  "Avg policy loss: {}".format(np.mean(p_losses)),
                  "Avg value loss: {}".format(np.mean(v_losses))]))
            logging.info(" ".join([
                  "Training thread: {}".format(rank),
                  "Num iters: {}K".format(num_iters),
                  "Avg policy loss: {}".format(np.mean(p_losses)),
                  "Avg value loss: {}".format(np.mean(v_losses))]))
            p_losses = []
            v_losses = []

        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
