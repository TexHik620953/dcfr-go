#python -m grpc_tools.protoc -I./dcfr-go/proto/infra/ --python_out=./ --pyi_out=./ --grpc_python_out=./ ./dcfr-go/proto/infra/actor.proto

import grpc
from concurrent import futures

from numpy.ma.extras import average
from torch.utils.tensorboard import SummaryWriter

import actor_pb2 as actor_pb2
import actor_pb2_grpc as actor_pb2_grpc
from networks.network import PokerStrategyNet
from utils.convert import convert_pbstate_to_tensor, convert_states_to_batch
import torch.nn.functional as F

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1 нейросеть для стратегии
avg_network = PokerStrategyNet("avg").to(device)
#avg_network.load()
avg_optimizer = torch.optim.Adam(avg_network.parameters(), lr=4e-4, weight_decay=2e-5)

# 3 нейросети для игроков
ply_networks = [PokerStrategyNet(f"ply{i}").to(device) for i in range(3)]
for pl_network in ply_networks:
    pl_network.load_state_dict(avg_network.state_dict())
    #pl_network.load()


ply_optimizers = [torch.optim.Adam(ply_networks[i].parameters(), lr=4e-4, weight_decay=2e-5) for i in range(3)]

tensorboard = SummaryWriter(log_dir="./tensorboard")
train_step = 0

def update_ema(beta=0.995):
    for avg_param, *ply_params in zip(
        avg_network.parameters(),
        *[net.parameters() for net in ply_networks]
    ):
        ply_mean = torch.stack([p.data for p in ply_params]).mean(dim=0)
        avg_param.data.mul_(beta).add_(ply_mean, alpha=1 - beta)


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

    def __train(self, network, optimizer, samples, name):
        global train_step
        for _ in range(1):
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
             ), (weights, iterations, regrets) = batch

            optimizer.zero_grad()
            logits = network((public_cards,
             private_cards,
             stakes,
             actions_mask,
             player_pots,
             active_players_mask,
             stage,
             current_player
             ))


            # Нормализуем regrets, чтобы получить "обучающие веса"
            regret_weights = regrets / (regrets.sum(dim=1, keepdim=True) + 1e-6)
            # Loss: минимизируем расхождение между предсказанными logits и целевыми весами из regrets
            loss = F.cross_entropy(logits, regret_weights) * weights.mean()

            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # Чем выше, тем лучше
            entropy = entropy.mean()
            loss = loss - 0.1 * entropy  # Коэффициент 0.1 можно настраивать


            loss.backward()
            optimizer.step()


            tensorboard.add_histogram(f"{name}/logits", logits.detach().cpu(), train_step)
            tensorboard.add_histogram(f"{name}/probs", probs.detach().cpu(), train_step)
            tensorboard.add_histogram(f"{name}/argmax", probs.detach().argmax(dim=1).cpu(), train_step)
            tensorboard.add_scalar(f"{name}/regrets", regrets.sum().cpu().item(), train_step)
            tensorboard.add_scalar(f"{name}/entropy", entropy.cpu().item(), train_step)

        network.save()
        return loss.item()

    def Train(self, request, context):
        if len(request.samples) > 0:
            curr_player = request.current_player
            net = ply_networks[curr_player]
            opt = ply_optimizers[curr_player]
            loss = self.__train(net, opt, request.samples, f"ply{curr_player}")
            tensorboard.add_scalar(f"loss/ply{curr_player}", loss, train_step)


            return actor_pb2.TrainResponse(loss=loss)
        return actor_pb2.TrainResponse(loss=0)

    def TrainAvg(self, request, context):
        update_ema()
        avg_network.save()
        return actor_pb2.TrainResponse(loss=0)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    actor_pb2_grpc.add_ActorServicer_to_server(ActorServicer(), server)
    server.add_insecure_port("[::]:1338")
    print("Ready")
    server.start()
    server.wait_for_termination()

serve()