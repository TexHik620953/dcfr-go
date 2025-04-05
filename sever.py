#python -m grpc_tools.protoc -I./dcfr-go/proto/infra/ --python_out=./ --pyi_out=./ --grpc_python_out=./ ./dcfr-go/proto/infra/actor.proto

import grpc
from concurrent import futures

from numpy.ma.extras import average
from torch.utils.tensorboard import SummaryWriter

import actor_pb2 as actor_pb2
import actor_pb2_grpc as actor_pb2_grpc
from networks.network import DeepCFRModel
from utils.convert import convert_pbstate_to_tensor, convert_states_to_batch
import torch.nn.functional as F

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1 нейросеть для стратегии
avg_network = DeepCFRModel("avg", lr=1e-3).to(device)
#avg_network.load()


initial_weigts = avg_network.state_dict()

# 3 нейросети для игроков
ply_networks = []
for i in range(3):
    net = DeepCFRModel(f"ply{i}", lr=1e-3).to(device)
    ply_networks.append(net)
    #net.load()
    net.load_state_dict(initial_weigts)


tensorboard = SummaryWriter(log_dir="./tensorboard")
train_step = 0


def train_net(network, samples):
    global train_step

    train_step += 1
    batch = convert_states_to_batch(samples, device)

    (public_cards,
     private_cards,
     stakes,
     actions_mask,
     player_pots,
     active_players_mask,
     stage,
     current_player
     ), (reach_prob, iterations, regrets, strategy) = batch

    network.optimizer.zero_grad()

    logits = network((public_cards,
                    private_cards,
                    stakes,
                    actions_mask,
                    player_pots,
                    active_players_mask,
                    stage,
                    current_player
    ))

    it_weights = (iterations + 1) / (iterations.max() + 1)
    loss = ((torch.square(logits - regrets)).sum(dim=1) * it_weights).mean()

    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # Чем выше, тем лучше
    entropy = entropy.mean()
    loss = loss - 0.005 * entropy

    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
    network.optimizer.step()

    tensorboard.add_histogram(f"{network.name}/logits", logits.detach().cpu(), train_step)
    tensorboard.add_histogram(f"{network.name}/argmax", logits.detach().argmax(dim=1).cpu(), train_step)
    tensorboard.add_scalar(f"{network.name}/regrets", regrets.sum(dim=1).mean().item(), train_step)
    # tensorboard.add_scalar(f"{network.name}/entropy", entropy.cpu().item(), train_step)

    return loss.item()

def train_avg_net(samples):
    global train_step
    train_step += 1

    batch = convert_states_to_batch(samples, device)

    (public_cards,
     private_cards,
     stakes,
     actions_mask,
     player_pots,
     active_players_mask,
     stage,
     current_player
     ), (reach_prob, iterations, regrets, strategy) = batch

    avg_network.optimizer.zero_grad()

    logits = avg_network((public_cards,
                      private_cards,
                      stakes,
                      actions_mask,
                      player_pots,
                      active_players_mask,
                      stage,
                      current_player
                      ))

    probs = F.softmax(logits, dim=1)

    it_weights = (iterations + 1) / (iterations.max() + 1)
    loss = ((torch.square(probs - strategy)).sum(dim=1) * it_weights).mean()

    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # Чем выше, тем лучше
    entropy = entropy.mean()
    loss = loss - 0.005 * entropy

    loss.backward()
    torch.nn.utils.clip_grad_norm_(avg_network.parameters(), 1.0)
    avg_network.optimizer.step()

    tensorboard.add_histogram(f"{avg_network.name}/logits", logits.detach().cpu(), train_step)
    tensorboard.add_histogram(f"{avg_network.name}/argmax", logits.detach().argmax(dim=1).cpu(), train_step)
    return loss.item()

class ActorServicer(actor_pb2_grpc.ActorServicer):
    def __init__(self):
        self.handled = 0
        self.avg_handled = 0
    def GetProbs(self, request, context):
        self.handled+=len(request.state)
        print(f"Handled: {self.handled} requests")
        curr_player = request.state[0].current_player

        state, actions_mask = convert_pbstate_to_tensor(request.state, device)
        probs = ply_networks[curr_player].get_probs(state, actions_mask).cpu().numpy()

        resp = actor_pb2.ActionProbsResponse()
        for unit_id in range(probs.shape[0]):
            r = actor_pb2.ProbsResponse()
            for i, prob in enumerate(probs[unit_id].tolist()):
                if prob > 1e-6:
                    r.action_probs[i] = prob
            resp.responses.append(r)
        return resp

    def GetAvgProbs(self, request, context):
        self.avg_handled+=len(request.state)
        print(f"Handled: {self.avg_handled} avg requests")

        state, actions_mask = convert_pbstate_to_tensor(request.state, device)
        probs = avg_network.get_probs(state, actions_mask).cpu().numpy()

        resp = actor_pb2.ActionProbsResponse()
        for unit_id in range(probs.shape[0]):
            r = actor_pb2.ProbsResponse()
            for i, prob in enumerate(probs[unit_id].tolist()):
                if prob > 1e-6:
                    r.action_probs[i] = prob
            resp.responses.append(r)
        return resp

    def Train(self, request, context):
        curr_player = request.current_player

        net = ply_networks[curr_player]
        loss = train_net(net, request.samples)
        tensorboard.add_scalar(f"loss/ply{curr_player}", loss, train_step)
        return actor_pb2.TrainResponse(loss=loss)

    def TrainAvg(self, request, context):
        loss = train_avg_net(request.samples)
        tensorboard.add_scalar(f"loss/avg", loss, train_step)
        return actor_pb2.TrainResponse(loss=loss)

    def Save(self, request, context):
        print("Saving network")
        avg_network.save()
        for net in ply_networks:
            net.save()
        return actor_pb2.Empty()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    actor_pb2_grpc.add_ActorServicer_to_server(ActorServicer(), server)
    server.add_insecure_port("[::]:1338")
    print("Ready")
    server.start()
    server.wait_for_termination()

serve()