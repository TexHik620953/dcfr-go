# python -m grpc_tools.protoc -I../dcfr-go/proto/infra/ --python_out=./ --pyi_out=./ --grpc_python_out=./ ../dcfr-go/proto/infra/actor.proto
# docker run -d -v /run/media/texhik/WORK/CODING/NeuralNetworks/dcfr-go/neural/tensorboard/:/app/runs/:ro -p 6006:6006 --name "my_tensorboard" schafo/tensorboard:latest --logdir=/app/runs --host 0.0.0.0
import time
import struct
import threading

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
checkpoint = "1773393125"

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
    net = AvgStrategyModel(f"avg{i}", lr=3e-3, hidden_dim=HIDDEN_DIM).to(device)
    avg_networks.append(net)
print("Avg strategy networks created")

for net in avg_networks:
    try:
        net.load(checkpoint, map_location=device)
        print(f"Loaded {net.name} from checkpoint {checkpoint}")
    except Exception as e:
        print(f"Could not load {net.name} checkpoint: {e}, using fresh weights")

tensorboard = SummaryWriter(log_dir="./tensorboard")


# ===== Binary file reader for TrainDirect =====

HEADER_SIZE = 64

def make_record_dtype(hidden_dim):
    """Build numpy structured dtype matching Go's binary record layout."""
    return np.dtype([
        ('active_players_mask', np.int32, 3),
        ('players_pots', np.int32, 3),
        ('stakes', np.int32, 3),
        ('legal_actions', np.float32, 10),
        ('stage', np.int32),
        ('current_player', np.int32),
        ('public_cards', np.int32, 5),
        ('private_cards', np.int32, 2),
        ('regrets', np.float32, 10),
        ('iteration', np.int32),
        ('context_h', np.float32, hidden_dim),
    ])


def read_header(path):
    """Read binary buffer header: max_samples, hidden_dim, counts[3]."""
    with open(path, 'rb') as f:
        hdr = f.read(HEADER_SIZE)
    magic = hdr[0:4]
    if magic != b'DCFR':
        raise ValueError(f"Invalid magic: {magic}")
    max_samples = struct.unpack('<i', hdr[8:12])[0]
    hidden_dim = struct.unpack('<i', hdr[12:16])[0]
    counts = [struct.unpack('<i', hdr[20 + i * 4:24 + i * 4])[0] for i in range(3)]
    return max_samples, hidden_dim, counts


def load_batch_from_mmap(mmap_data, player_id, count, max_samples, batch_size, rng):
    """Load a random batch from the memory-mapped binary file.
    Returns numpy structured array of shape (batch_size,)."""
    indices = rng.integers(0, count, size=batch_size)
    offset = player_id * max_samples
    return mmap_data[offset + indices].copy()  # copy to avoid holding mmap pages


def batch_to_tensors(batch, hidden_dim):
    """Convert numpy structured batch to GPU tensors. Returns (state_tuple, regrets, iterations, context_h)."""
    # Convert to float32 tensors
    active_mask = torch.as_tensor(batch['active_players_mask'].astype(np.float32), device=device)
    pots = torch.as_tensor(batch['players_pots'].astype(np.float32), device=device)
    stakes = torch.as_tensor(batch['stakes'].astype(np.float32), device=device)
    actions_mask = torch.as_tensor(batch['legal_actions'], device=device)
    stage = torch.as_tensor(batch['stage'].reshape(-1, 1), device=device)
    current_player = torch.as_tensor(batch['current_player'].reshape(-1, 1), device=device)
    public_cards = torch.as_tensor(batch['public_cards'], device=device)
    private_cards = torch.as_tensor(batch['private_cards'], device=device)
    regrets = torch.as_tensor(batch['regrets'], device=device)
    iterations = torch.as_tensor(batch['iteration'].astype(np.float32), device=device)
    context_h = torch.as_tensor(batch['context_h'], device=device)

    # Normalize pots/stakes by total bank (same as convert.py)
    bank = stakes.sum(dim=1, keepdim=True) + pots.sum(dim=1, keepdim=True)
    bank = bank.clamp(min=1e-8)
    stakes = stakes / bank
    pots = pots / bank

    state_tuple = (public_cards, private_cards, stakes, actions_mask, pots, active_mask, stage, current_player)
    return state_tuple, regrets, iterations, context_h


def train_step_direct(network, state_tuple, regrets, iterations, context_h):
    """Single training step using pre-converted tensors."""
    stages = state_tuple[6].squeeze(1)

    network.optimizer.zero_grad()
    features = network.encode_features(state_tuple)
    new_context = network.context_updater(features, context_h)
    logits = network.get_action_logits(features, new_context, stages)

    dcfr_weights = (iterations + 1).pow(DCFR_ALPHA)
    dcfr_weights = dcfr_weights / dcfr_weights.sum()

    loss = ((torch.square(logits - regrets)).sum(dim=1) * dcfr_weights).sum()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), 2)
    network.optimizer.step()

    total_loss = loss.item()

    if network.step % 2 == 0:
        step = network.step
        tensorboard.add_scalar(f"{network.name}/total_loss", total_loss, step)
        tensorboard.add_scalar(f"{network.name}/learning_rate",
                               network.optimizer.param_groups[0]['lr'], step)
        if step % 10 == 0:
            total_grad_norm = sum(
                p.grad.norm().item() ** 2 for p in network.parameters() if p.grad is not None
            )
            tensorboard.add_scalar(f"{network.name}/grad_total_norm",
                                   total_grad_norm ** 0.5, step)

    network.step += 1
    return total_loss


def prefetch_batch(mmap_data, player_id, count, max_samples, batch_size, hidden_dim, rng):
    """Load batch from mmap and convert to tensors in a background thread."""
    batch = load_batch_from_mmap(mmap_data, player_id, count, max_samples, batch_size, rng)
    return batch_to_tensors(batch, hidden_dim)


def train_direct_loop(player_id, batch_size, iterations, db_path, max_samples_hint):
    """Run full training loop: read binary file, prefetch, train."""
    t0 = time.perf_counter()

    max_samples, hidden_dim, counts = read_header(db_path)
    count = counts[player_id]
    if count == 0:
        print(f"[TrainDirect] P{player_id}: no samples")
        return 0.0

    record_dtype = make_record_dtype(hidden_dim)
    total_records = 3 * max_samples

    # Memory-map the entire records section (read-only)
    mmap_data = np.memmap(db_path, dtype=record_dtype, mode='r',
                          offset=HEADER_SIZE, shape=(total_records,))

    network = ply_networks[player_id]
    rng = np.random.default_rng()

    # Prefetch first batch in background
    prefetch_result = [None]
    prefetch_event = threading.Event()

    def do_prefetch():
        prefetch_result[0] = prefetch_batch(
            mmap_data, player_id, count, max_samples, batch_size, hidden_dim, rng
        )
        prefetch_event.set()

    thread = threading.Thread(target=do_prefetch)
    thread.start()

    total_loss = 0.0
    valid_iters = 0

    for i in range(iterations):
        # Wait for current batch
        prefetch_event.wait()
        prefetch_event.clear()
        state_tuple, regrets, iters_t, context_h = prefetch_result[0]

        # Start prefetching next batch (if not last iteration)
        if i < iterations - 1:
            thread = threading.Thread(target=do_prefetch)
            thread.start()

        # Train on current batch
        loss = train_step_direct(network, state_tuple, regrets, iters_t, context_h)
        total_loss += loss
        valid_iters += 1

    # Wait for any outstanding prefetch thread
    if thread.is_alive():
        thread.join()

    del mmap_data  # release mmap

    elapsed = time.perf_counter() - t0
    avg_loss = total_loss / valid_iters if valid_iters > 0 else 0.0
    print(f"[TrainDirect] P{player_id}: {valid_iters} iters, avg_loss={avg_loss:.6f}, "
          f"samples={count}, elapsed={elapsed:.1f}s")
    return avg_loss


# ===== Legacy training functions (kept for old Train RPC) =====

def train_net(network, game_samples):
    """Train advantage network. Each sample is independent with its saved context vector."""
    flat_samples = [sample for game in game_samples for sample in game.samples]

    if len(flat_samples) == 0:
        return 0.0

    samples, (iterations, regrets) = convert_states_to_batch(flat_samples, device)
    stages = samples[6].squeeze(1)

    network.optimizer.zero_grad()

    features = network.encode_features(samples)  # [N, hidden]

    # Reconstruct fixed-size context vectors from saved state (vectorized)
    hidden_dim = network.hidden_dim
    ctx_np = np.zeros((len(flat_samples), hidden_dim), dtype=np.float32)
    for i, sample in enumerate(flat_samples):
        h_flat = sample.lstm_context_h
        if len(h_flat) == hidden_dim:
            ctx_np[i] = h_flat
    context = torch.as_tensor(ctx_np, device=device)  # [N, hidden]

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
    flat_samples = [sample for game in game_samples for sample in game.samples]

    if len(flat_samples) == 0:
        return 0.0

    samples, (iterations, target_strategies) = convert_strategy_states_to_batch(flat_samples, device)
    stages = samples[6].squeeze(1)

    network.optimizer.zero_grad()

    features = network.encode_features(samples)

    hidden_dim = network.hidden_dim
    ctx_np = np.zeros((len(flat_samples), hidden_dim), dtype=np.float32)
    for i, sample in enumerate(flat_samples):
        h_flat = sample.lstm_context_h
        if len(h_flat) == hidden_dim:
            ctx_np[i] = h_flat
    context = torch.as_tensor(ctx_np, device=device)

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

    def TrainDirect(self, request, context):
        try:
            avg_loss = train_direct_loop(
                player_id=request.current_player,
                batch_size=request.batch_size,
                iterations=request.iterations,
                db_path=request.db_path,
                max_samples_hint=request.max_samples,
            )
            return actor_pb2.TrainDirectResponse(avg_loss=avg_loss)
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
