#python -m grpc_tools.protoc -I./dcfr-go/proto/infra/ --python_out=./ --pyi_out=./ --grpc_python_out=./ ./dcfr-go/proto/infra/actor.proto
#docker run -d -v /run/media/texhik/WORK/CODING/NeuralNetworks/dcfr-go/neural/tensorboard/:/app/runs/:ro -p 6006:6006 --name "my_tensorboard" schafo/tensorboard:latest --logdir=/app/runs --host 0.0.0.0

print("Init")
import grpc
from concurrent import futures

from numpy.ma.extras import average
from torch.utils.tensorboard import SummaryWriter

import actor_pb2 as actor_pb2
import actor_pb2_grpc as actor_pb2_grpc
from networks.network import DeepCFRModel
from utils.convert import convert_pbstate_to_tensor, convert_states_to_batch
import torch.nn.functional as F
import datetime
import torch
print("Init")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Launching on: ", device)
# 3 нейросети для игроков

# Создаем игроков
ply_networks = []
for i in range(3):
    print("Creating: ", i, " network")
    net = DeepCFRModel(f"ply{i}", lr=1e-3).to(device)
    ply_networks.append(net)

# Пытаемся загрузить оригинальную сеть
try:
    ply_networks[0].load("initial")
    print("Initial model loaded")
except:
    print("Initial model not found, initializing...")
    ply_networks[0].save("initial")

initial_state = ply_networks[0].state_dict()
intial_opt_state = ply_networks[0].optimizer.state_dict()

for net in ply_networks[1:]:
    net.load_state_dict(initial_state)
    net.optimizer.load_state_dict(intial_opt_state)
print("Networks created")

tensorboard = SummaryWriter(log_dir="./tensorboard")


def train_net(network, samples):
    batch = convert_states_to_batch(samples, device)

    (public_cards,
     private_cards,
     stakes,
     actions_mask,
     player_pots,
     active_players_mask,
     stage,
     current_player
     ), (reach_prob, iterations, regrets) = batch

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

    # Linear CFR weighting: iteration t gets weight proportional to t
    it_weights = (iterations + 1) / (iterations.max() + 1)

    # MSE loss between predicted advantages and actual regrets (no clamping)
    loss = ((torch.square(logits - regrets)).sum(dim=1) * it_weights).mean()

    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # Чем выше, тем лучше
    entropy = entropy.mean()
    loss = loss - 0.005 * entropy

    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), 2)
    network.optimizer.step()
    network.scheduler.step()
    if network.step % 50 == 0:
        tensorboard.add_histogram(f"{network.name}/logits", logits.detach().cpu(), network.step)
        tensorboard.add_histogram(f"{network.name}/argmax", logits.detach().argmax(dim=1).cpu(), network.step)
        tensorboard.add_scalar(f"{network.name}/regrets", regrets.sum(dim=1).mean().item(), network.step)
        tensorboard.add_scalar(f"{network.name}/loss", loss.item(), network.step)
    network.step+=1
    
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
                if prob > 0:
                    r.action_probs[i] = prob
            resp.responses.append(r)
        return resp

    def Train(self, request, context):
        curr_player = request.current_player

        net = ply_networks[curr_player]
        loss = train_net(net, request.samples)
        return actor_pb2.TrainResponse(loss=loss)

    def Save(self, request, context):
        print("Saving networks")
        for net in ply_networks:
            net.save(datetime.datetime.now())
        return actor_pb2.Empty()

    def Reset(self, request, context):
        global initial_state
        print("Resetting networks")
        for net in ply_networks:
            net.load_state_dict(initial_state)
            net.optimizer.load_state_dict(intial_opt_state)
        return actor_pb2.Empty()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    actor_pb2_grpc.add_ActorServicer_to_server(ActorServicer(), server)
    server.add_insecure_port("0.0.0.0:1338")
    print("Ready")
    server.start()
    server.wait_for_termination()

serve()