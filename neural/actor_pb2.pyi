from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GameStage(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PREFLOP: _ClassVar[GameStage]
    FLOP: _ClassVar[GameStage]
    TURN: _ClassVar[GameStage]
    RIVER: _ClassVar[GameStage]
    SHOWDOWN: _ClassVar[GameStage]
PREFLOP: GameStage
FLOP: GameStage
TURN: GameStage
RIVER: GameStage
SHOWDOWN: GameStage

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GameState(_message.Message):
    __slots__ = ("active_players_mask", "players_pots", "stakes", "legal_actions", "stage", "current_player", "public_cards", "private_cards")
    class LegalActionsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: bool
        def __init__(self, key: _Optional[int] = ..., value: bool = ...) -> None: ...
    ACTIVE_PLAYERS_MASK_FIELD_NUMBER: _ClassVar[int]
    PLAYERS_POTS_FIELD_NUMBER: _ClassVar[int]
    STAKES_FIELD_NUMBER: _ClassVar[int]
    LEGAL_ACTIONS_FIELD_NUMBER: _ClassVar[int]
    STAGE_FIELD_NUMBER: _ClassVar[int]
    CURRENT_PLAYER_FIELD_NUMBER: _ClassVar[int]
    PUBLIC_CARDS_FIELD_NUMBER: _ClassVar[int]
    PRIVATE_CARDS_FIELD_NUMBER: _ClassVar[int]
    active_players_mask: _containers.RepeatedScalarFieldContainer[int]
    players_pots: _containers.RepeatedScalarFieldContainer[int]
    stakes: _containers.RepeatedScalarFieldContainer[int]
    legal_actions: _containers.ScalarMap[int, bool]
    stage: GameStage
    current_player: int
    public_cards: _containers.RepeatedScalarFieldContainer[int]
    private_cards: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, active_players_mask: _Optional[_Iterable[int]] = ..., players_pots: _Optional[_Iterable[int]] = ..., stakes: _Optional[_Iterable[int]] = ..., legal_actions: _Optional[_Mapping[int, bool]] = ..., stage: _Optional[_Union[GameStage, str]] = ..., current_player: _Optional[int] = ..., public_cards: _Optional[_Iterable[int]] = ..., private_cards: _Optional[_Iterable[int]] = ...) -> None: ...

class StateSample(_message.Message):
    __slots__ = ("game_state", "regrets", "iteration", "lstm_context_h", "lstm_context_c")
    class RegretsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: float
        def __init__(self, key: _Optional[int] = ..., value: _Optional[float] = ...) -> None: ...
    GAME_STATE_FIELD_NUMBER: _ClassVar[int]
    REGRETS_FIELD_NUMBER: _ClassVar[int]
    ITERATION_FIELD_NUMBER: _ClassVar[int]
    LSTM_CONTEXT_H_FIELD_NUMBER: _ClassVar[int]
    LSTM_CONTEXT_C_FIELD_NUMBER: _ClassVar[int]
    game_state: GameState
    regrets: _containers.ScalarMap[int, float]
    iteration: int
    lstm_context_h: _containers.RepeatedScalarFieldContainer[float]
    lstm_context_c: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, game_state: _Optional[_Union[GameState, _Mapping]] = ..., regrets: _Optional[_Mapping[int, float]] = ..., iteration: _Optional[int] = ..., lstm_context_h: _Optional[_Iterable[float]] = ..., lstm_context_c: _Optional[_Iterable[float]] = ...) -> None: ...

class GameSample(_message.Message):
    __slots__ = ("samples",)
    SAMPLES_FIELD_NUMBER: _ClassVar[int]
    samples: _containers.RepeatedCompositeFieldContainer[StateSample]
    def __init__(self, samples: _Optional[_Iterable[_Union[StateSample, _Mapping]]] = ...) -> None: ...

class CFRState(_message.Message):
    __slots__ = ("game_state", "lstm_context_h", "lstm_context_c")
    GAME_STATE_FIELD_NUMBER: _ClassVar[int]
    LSTM_CONTEXT_H_FIELD_NUMBER: _ClassVar[int]
    LSTM_CONTEXT_C_FIELD_NUMBER: _ClassVar[int]
    game_state: GameState
    lstm_context_h: _containers.RepeatedScalarFieldContainer[float]
    lstm_context_c: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, game_state: _Optional[_Union[GameState, _Mapping]] = ..., lstm_context_h: _Optional[_Iterable[float]] = ..., lstm_context_c: _Optional[_Iterable[float]] = ...) -> None: ...

class ActionProbsRequest(_message.Message):
    __slots__ = ("states",)
    STATES_FIELD_NUMBER: _ClassVar[int]
    states: _containers.RepeatedCompositeFieldContainer[CFRState]
    def __init__(self, states: _Optional[_Iterable[_Union[CFRState, _Mapping]]] = ...) -> None: ...

class ActionProbsResponse(_message.Message):
    __slots__ = ("responses",)
    RESPONSES_FIELD_NUMBER: _ClassVar[int]
    responses: _containers.RepeatedCompositeFieldContainer[ProbsResponse]
    def __init__(self, responses: _Optional[_Iterable[_Union[ProbsResponse, _Mapping]]] = ...) -> None: ...

class ProbsResponse(_message.Message):
    __slots__ = ("action_probs", "lstm_context_h", "lstm_context_c")
    class ActionProbsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: float
        def __init__(self, key: _Optional[int] = ..., value: _Optional[float] = ...) -> None: ...
    ACTION_PROBS_FIELD_NUMBER: _ClassVar[int]
    LSTM_CONTEXT_H_FIELD_NUMBER: _ClassVar[int]
    LSTM_CONTEXT_C_FIELD_NUMBER: _ClassVar[int]
    action_probs: _containers.ScalarMap[int, float]
    lstm_context_h: _containers.RepeatedScalarFieldContainer[float]
    lstm_context_c: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, action_probs: _Optional[_Mapping[int, float]] = ..., lstm_context_h: _Optional[_Iterable[float]] = ..., lstm_context_c: _Optional[_Iterable[float]] = ...) -> None: ...

class TrainRequest(_message.Message):
    __slots__ = ("current_player", "game_samples")
    CURRENT_PLAYER_FIELD_NUMBER: _ClassVar[int]
    GAME_SAMPLES_FIELD_NUMBER: _ClassVar[int]
    current_player: int
    game_samples: _containers.RepeatedCompositeFieldContainer[GameSample]
    def __init__(self, current_player: _Optional[int] = ..., game_samples: _Optional[_Iterable[_Union[GameSample, _Mapping]]] = ...) -> None: ...

class TrainResponse(_message.Message):
    __slots__ = ("loss",)
    LOSS_FIELD_NUMBER: _ClassVar[int]
    loss: float
    def __init__(self, loss: _Optional[float] = ...) -> None: ...
