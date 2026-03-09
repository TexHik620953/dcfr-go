# python -m grpc_tools.protoc -I../dcfr-go/proto/infra/ --python_out=./ --pyi_out=./ --grpc_python_out=./ ../dcfr-go/proto/infra/actor.proto
# docker run -d -v /run/media/texhik/WORK/CODING/NeuralNetworks/dcfr-go/neural/tensorboard/:/app/runs/:ro -p 6006:6006 --name "my_tensorboard" schafo/tensorboard:latest --logdir=/app/runs --host 0.0.0.0
import time

print("Init")
import grpc
from concurrent import futures
import numpy as np
import traceback

from torch.utils.tensorboard import SummaryWriter

import actor_pb2 as actor_pb2
import actor_pb2_grpc as actor_pb2_grpc
from networks.network import DeepCFRModel, AvgStrategyModel, NUM_ACTIONS
from utils.convert import convert_pbstate_to_tensor, convert_states_to_batch, convert_strategy_states_to_batch
import torch.nn.functional as F
import datetime
import torch
print("Init")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Launching on: ", device)

# DCFR weighting parameter
DCFR_ALPHA = 1.5
HIDDEN_DIM = 256
checkpoint = "1773048994"

# Create player networks
ply_networks = []
for i in range(3):
    print("Creating: ", i, " network")
    net = DeepCFRModel(f"ply{i}", lr=1e-3, hidden_dim=HIDDEN_DIM).to(device)
    ply_networks.append(net)

# Try to load checkpoint
for net in ply_networks:
    try:
        net.load(checkpoint, map_location=device)
        print(f"Loaded {net.name} from checkpoint {checkpoint}")
    except Exception as e:
        print(f"Could not load {net.name} checkpoint: {e}, using fresh weights")

# Create average strategy networks (one per player)
avg_networks = []
for i in range(3):
    print("Creating avg strategy network: ", i)
    net = AvgStrategyModel(f"avg{i}", lr=1e-3, hidden_dim=HIDDEN_DIM).to(device)
    avg_networks.append(net)
print("Avg strategy networks created")

for net in avg_networks:
    try:
        net.load(checkpoint, map_location=device)
        print(f"Loaded {net.name} from checkpoint {checkpoint}")
    except Exception as e:
        print(f"Could not load {net.name} checkpoint: {e}, using fresh weights")

tensorboard = SummaryWriter(log_dir="./tensorboard")


def train_net(network, game_samples):
    """Train advantage network. Each sample is independent with its saved context vector."""
    flat_samples = []
    for game in game_samples:
        for sample in game.samples:
            flat_samples.append(sample)

    if len(flat_samples) == 0:
        return 0.0

    samples, (iterations, regrets) = convert_states_to_batch(flat_samples, device)
    stages = samples[6].squeeze(1)

    network.optimizer.zero_grad()

    features = network.encode_features(samples)  # [N, hidden]

    # Reconstruct fixed-size context vectors from saved state
    hidden_dim = network.hidden_dim
    contexts = []
    for sample in flat_samples:
        h_flat = list(sample.lstm_context_h)
        if len(h_flat) == hidden_dim:
            contexts.append(torch.tensor(h_flat, device=device, dtype=torch.float32))
        else:
            contexts.append(torch.zeros(hidden_dim, device=device))
    context = torch.stack(contexts)  # [N, hidden]

    # Update context through GRU
    new_context = network.context_updater(features, context)

    logits = network.get_action_logits(features, new_context, stages)

    # === DCFR weighting: w_t = t^alpha ===
    dcfr_weights = (iterations + 1).pow(DCFR_ALPHA)
    dcfr_weights = dcfr_weights / dcfr_weights.sum()

    # MSE loss with DCFR weighting (no regret normalization — preserves per-state signal)
    loss = ((torch.square(logits - regrets)).sum(dim=1) * dcfr_weights).sum()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), 2)
    network.optimizer.step()
    network.scheduler.step()

    total_loss = loss.item()

    # TensorBoard logging
    if network.step % 2 == 0:
        step = network.step
        tensorboard.add_scalar(f"{network.name}/total_loss", total_loss, step)
        tensorboard.add_scalar(f"{network.name}/learning_rate",
                               network.optimizer.param_groups[0]['lr'], step)

        if step % 10 == 0:
            total_grad_norm = 0
            for name, param in network.named_parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item() ** 2
            tensorboard.add_scalar(f"{network.name}/grad_total_norm",
                                   total_grad_norm ** 0.5, step)

    network.step += 1
    return total_loss


def train_avg_net(network, game_samples):
    """Train average strategy network using KL divergence loss."""
    flat_samples = []
    for game in game_samples:
        for sample in game.samples:
            flat_samples.append(sample)

    if len(flat_samples) == 0:
        return 0.0

    samples, (iterations, target_strategies) = convert_strategy_states_to_batch(flat_samples, device)
    stages = samples[6].squeeze(1)

    network.optimizer.zero_grad()

    features = network.encode_features(samples)

    hidden_dim = network.hidden_dim
    contexts = []
    for sample in flat_samples:
        h_flat = list(sample.lstm_context_h)
        if len(h_flat) == hidden_dim:
            contexts.append(torch.tensor(h_flat, device=device, dtype=torch.float32))
        else:
            contexts.append(torch.zeros(hidden_dim, device=device))
    context = torch.stack(contexts)

    new_context = network.context_updater(features, context)

    logits = network.get_action_logits(features, new_context, stages)

    # Linear weighting for average strategy: w_t = t + 1
    linear_weights = (iterations + 1)
    linear_weights = linear_weights / linear_weights.sum()

    log_probs = F.log_softmax(logits, dim=1)
    target_clamped = target_strategies.clamp(min=1e-8)
    kl_loss = F.kl_div(log_probs, target_clamped, reduction='none').sum(dim=1)
    loss = (kl_loss * linear_weights).sum()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), 2)
    network.optimizer.step()
    network.scheduler.step()

    total_loss = loss.item()

    if network.step % 2 == 0:
        tensorboard.add_scalar(f"{network.name}/total_loss", total_loss, network.step)

    network.step += 1
    return total_loss


class ActorServicer(actor_pb2_grpc.ActorServicer):
    def __init__(self):
        self.handled = 0

    def GetProbs(self, request, context):
        try:
            import time as _time
            t0 = _time.perf_counter()

            batch_size = len(request.states)
            self.handled += batch_size
            curr_player = request.states[0].game_state.current_player

            t1 = _time.perf_counter()
            state, actions_mask, history_h = convert_pbstate_to_tensor(request.states, device)

            # Reconstruct fixed-size context vectors — batch at once
            hidden_dim = ply_networks[curr_player].hidden_dim
            prev_context = None
            if history_h[0] is not None:
                ctx_np = np.zeros((batch_size, hidden_dim), dtype=np.float32)
                for idx, h_flat in enumerate(history_h):
                    if h_flat is not None and len(h_flat) == hidden_dim:
                        ctx_np[idx] = h_flat
                prev_context = torch.as_tensor(ctx_np, device=device)

            t2 = _time.perf_counter()
            probs, new_context = ply_networks[curr_player].get_probs(state, actions_mask, prev_context)

            t3 = _time.perf_counter()
            probs_np = probs.cpu().numpy()
            context_np = new_context.cpu().numpy()

            t4 = _time.perf_counter()
            # Batch convert to Python lists (much faster than per-row .tolist())
            probs_list = probs_np.tolist()
            context_list = context_np.tolist()

            resp = actor_pb2.ActionProbsResponse()
            for unit_id in range(batch_size):
                r = actor_pb2.ProbsResponse()
                for i, prob in enumerate(probs_list[unit_id]):
                    if prob > 1e-8:
                        r.action_probs[i] = prob
                r.lstm_context_h[:] = context_list[unit_id]
                resp.responses.append(r)

            t5 = _time.perf_counter()
            if self.handled % 100000 < batch_size:
                print(f"[GetProbs] batch={batch_size} | "
                      f"parse={t2-t1:.3f}s | inference={t3-t2:.3f}s | "
                      f"to_cpu={t4-t3:.3f}s | build_resp={t5-t4:.3f}s | "
                      f"total={t5-t0:.3f}s")
            return resp
        except Exception as e:
            traceback.print_exc()
            raise e

    def Train(self, request, context):
        try:
            curr_player = request.current_player
            net = ply_networks[curr_player]
            loss = train_net(net, request.game_samples)
            return actor_pb2.TrainResponse(loss=loss)
        except Exception as e:
            traceback.print_exc()
            raise e

    def TrainAvgStrategy(self, request, context):
        try:
            curr_player = request.current_player
            net = avg_networks[curr_player]
            loss = train_avg_net(net, request.game_samples)
            return actor_pb2.TrainResponse(loss=loss)
        except Exception as e:
            traceback.print_exc()
            raise e

    def Save(self, request, context):
        print("Saving networks")
        tt = int(time.time())
        for net in ply_networks:
            net.save(tt)
        for net in avg_networks:
            net.save(tt)
        return actor_pb2.Empty()

    def Reset(self, request, context):
        return actor_pb2.Empty()

def serve():
    server_options = [
        ('grpc.max_send_message_length', 512 * 1024 * 1024),
        ('grpc.max_receive_message_length', 512 * 1024 * 1024),
        ('grpc.max_metadata_size', 16 * 1024 * 1024),
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
