# python -m grpc_tools.protoc -I../dcfr-go/proto/infra/ --python_out=./ --pyi_out=./ --grpc_python_out=./ ../dcfr-go/proto/infra/actor.proto
# docker run -d -v /run/media/texhik/WORK/CODING/NeuralNetworks/dcfr-go/neural/tensorboard/:/app/runs/:ro -p 6006:6006 --name "my_tensorboard" schafo/tensorboard:latest --logdir=/app/runs --host 0.0.0.0

print("Init")
import grpc
from concurrent import futures
import numpy as np

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



def train_net(network, game_samples):
    data = [[sample for sample in game.samples] for game in game_samples]


    game_indexes = []
    game_samples = []

    for game_idx in range(len(data)):
        game = data[game_idx]
        for episode_idx in range(len(game)):
            eposode = game[episode_idx]
            game_samples.append(eposode)
            game_indexes.append(game_idx)

    samples, (iterations, regrets) = convert_states_to_batch(game_samples, device)

    # Start training here
    network.optimizer.zero_grad()

    features = network.encode_features(samples)
    # Now we have all features for all steps for all games
    # We need to recombine them back to steps

    # stages[game][stage]
    stages = [[]for _ in range(len(game_indexes))]
    for game_idx in game_indexes:
        stages[game_idx].append((features[game_idx], iterations[game_idx], regrets[game_idx]))

    # transpose list to stages[stage][game]
    # Longest games first
    stages.sort(key=len, reverse=True)
    longest_game = len(stages[0])

    transposed_stages = [[] for _ in range(longest_game)]
    for game in stages:
        for stage_idx in range(len(game)):
            stage = game[stage_idx]
            transposed_stages[stage_idx].append(stage)

    lstm_contexts = None

    total_loss = 0
    stage_losses = []
    stage_entropies = []
    stage_means = []
    stage_stds = []

    # Iterate over stages and calculate loss
    for stage_idx in range(len(transposed_stages)):
        stage = transposed_stages[stage_idx]

        features = torch.vstack([s[0] for s in stage])
        iterations = torch.vstack([s[1] for s in stage])
        regrets = torch.vstack([s[2] for s in stage])
        # Нормализуем регреты
        regrets = (regrets - regrets.mean(dim=1, keepdim=True))/regrets.std(dim=1, keepdim=True)

        # Cut lstm context from previous iteration
        if lstm_contexts is not None:
            lstm_contexts = (lstm_contexts[0][:, :features.shape[0],:], lstm_contexts[1][:, :features.shape[0],:])

        logits, new_context = network.process_features(features, lstm_contexts)

        lstm_contexts = new_context

        # Linear CFR weighting: iteration t gets weight proportional to t
        it_weights = (iterations + 1) / (iterations.max() + 1)

        # Calculate loss
        # MSE loss between predicted advantages and actual regrets (no clamping)
        loss = ((torch.square(logits - regrets)).sum(dim=1) * it_weights).mean()

        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # Чем выше, тем лучше
        entropy = entropy.mean()
        loss = loss - 0.005 * entropy
        total_loss += loss.item()
        loss.backward(retain_graph=(stage_idx<len(transposed_stages)-1))

        stage_losses.append(loss.item())
        stage_entropies.append(entropy.item())
        stage_means.append(regrets.mean().item())
        stage_stds.append(regrets.std().item())


    torch.nn.utils.clip_grad_norm_(network.parameters(), 2)
    network.optimizer.step()
    network.scheduler.step()

    # Логирование в TensorBoard
    if network.step % 10 == 0:  # Каждые 10 шагов
        step = network.step

        # Основные метрики
        tensorboard.add_scalar(f"{network.name}/total_loss", total_loss, step)
        tensorboard.add_scalar(f"{network.name}/avg_stage_loss", np.mean(stage_losses), step)
        tensorboard.add_scalar(f"{network.name}/avg_entropy", np.mean(stage_entropies), step)
        tensorboard.add_scalar(f"{network.name}/learning_rate",
                               network.optimizer.param_groups[0]['lr'], step)

        # Статистика регретов
        tensorboard.add_scalar(f"{network.name}/regret_mean", np.mean(stage_means), step)
        tensorboard.add_scalar(f"{network.name}/regret_std", np.mean(stage_stds), step)

        # Градиенты (каждые 50 шагов)
        if step % 10 == 0:
            total_grad_norm = 0
            for name, param in network.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm ** 2

                    # Логируем градиенты для важных слоев
                    if 'lstm' in name or 'attention' in name or 'head' in name:
                        tensorboard.add_histogram(f"{network.name}/grads/{name}",
                                                  param.grad.cpu(), step)
                        tensorboard.add_scalar(f"{network.name}/grad_norm/{name}",
                                               grad_norm, step)

            # Общая норма градиентов
            tensorboard.add_scalar(f"{network.name}/grad_total_norm",
                                   total_grad_norm ** 0.5, step)

            # Веса модели
            for name, param in network.named_parameters():
                if 'lstm' in name or 'attention' in name or 'head' in name:
                    tensorboard.add_histogram(f"{network.name}/weights/{name}",
                                              param.data.cpu(), step)

    network.step += 1
    return total_loss

class ActorServicer(actor_pb2_grpc.ActorServicer):
    def __init__(self):
        self.handled = 0
        self.avg_handled = 0
    def GetProbs(self, request, context):
        self.handled+=len(request.states)
        print(f"Handled: {self.handled} requests")
        curr_player = request.states[0].game_state.current_player

        state, actions_mask, lstm_context = convert_pbstate_to_tensor(request.states, device)
        probs, (lstm_h, lstm_c) = ply_networks[curr_player].get_probs(state, actions_mask, lstm_context)

        probs = probs.cpu().numpy()
        lstm_h = lstm_h.cpu().numpy()
        lstm_c = lstm_c.cpu().numpy()

        resp = actor_pb2.ActionProbsResponse()
        for unit_id in range(probs.shape[0]):
            r = actor_pb2.ProbsResponse()

            for i, prob in enumerate(probs[unit_id].tolist()):
                if prob > 1e-8:
                    r.action_probs[i] = prob
            r.lstm_context_h.extend(lstm_h[:, unit_id, :].flatten().tolist())
            r.lstm_context_c.extend(lstm_c[:, unit_id, :].flatten().tolist())

            if len(r.action_probs) == 0:
                print(r.action_probs)
            resp.responses.append(r)
        return resp

    def Train(self, request, context):
        curr_player = request.current_player

        net = ply_networks[curr_player]
        loss = train_net(net, request.game_samples)
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
    server_options = [
        ('grpc.max_send_message_length', 512 * 1024 * 1024),  # 512 MB
        ('grpc.max_receive_message_length', 512 * 1024 * 1024),  # 512 MB
        ('grpc.max_metadata_size', 16 * 1024 * 1024),  # Опционально: увеличить размер метаданных
    ]

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=server_options
    )
    actor_pb2_grpc.add_ActorServicer_to_server(ActorServicer(), server)
    server.add_insecure_port("0.0.0.0:1338")
    print("Ready")
    server.start()
    server.wait_for_termination()

serve()