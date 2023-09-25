import dataclasses
from dataclasses import dataclass, fields
from enum import Enum
import json


def from_dict(dict_:dict):
    for key in fields(Bag):
        if key.name not in dict_.keys():
            dict_[key.name] = None

    bag = Bag(
        bag=dict_["bag"],
        test_date=dict_["test_date"],
        test_label=dict_["test_label"],
        model_cell=dict_["model_cell"],
        model_grf=dict_["model_grf"],
        name=dict_["name"],
        body_weight=dict_["body_weight"],
        sole_size=dict_["sole_size"],
        paretic_side=dict_["paretic_side"],
        session_type=dict_["session_type"],
        distance=dict_["distance"],
        time=dict_["time"],
        bpm=dict_["bpm"],
        ignore_cycle=dict_["ignore_cycle"]
    )
    return bag


class Side(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class Session(str, Enum):
    TWO_MIN_OFF_CUE_OFF = "2MWT_OFF_CUE_OFF"
    TWO_MIN_OFF_CUE_ON = "2MWT_OFF_CUE_ON"
    TWO_MIN_ON_CUE_OFF = "2MWT_ON_CUE_OFF"
    TWO_MIN_ON_CUE_ON = "2MWT_ON_CUE_ON"
    TWO_MIN_BARE_CUE_OFF = "2MWT_BARE_CUE_OFF"
    TWO_MIN_BARE_CUE_ON = "2MWT_BARE_CUE_ON"
    TEN_METER_OFF_CUE_OFF = "10MWT_OFF_CUE_OFF"
    TEN_METER_OFF_CUE_ON = "10MWT_OFF_CUE_ON"
    TEN_METER_ON_CUE_OFF = "10MWT_ON_CUE_OFF"
    TEN_METER_ON_CUE_ON = "10MWT_ON_CUE_ON"
    TEN_METER_BARE_CUE_OFF = "10MWT_BARE_CUE_OFF"
    TEN_METER_BARE_CUE_ON = "10MWT_BARE_CUE_ON"


@dataclass
class Bag:
    bag: str
    test_date: str
    test_label: str
    model_cell: str
    model_grf: str
    name: str
    body_weight: float
    sole_size: int
    paretic_side: Side
    session_type: Session
    distance: float = None
    time: float = None
    bpm: tuple = (None, None)
    ignore_cycle: tuple = (None, None)


@dataclass
class ClinicalAnalysis:
    limb_length: dict
    grf_max: list
    grf_impulse: list
    toe_clearance: list
    stance_time: list
    gait_speed_imu: list
    gait_speed_distance: list
    data_bag: Bag

#
bag = Bag(
    bag='log_2023-08-16-15-56-59.bag',
    test_date='20230816',
    test_label="RH-23-01",
    model_cell="CHAR_230815_280_LP",
    model_grf="GRF_230815/LSTM_GRF.pt",
    name="KJJ",
    body_weight=69.7,
    sole_size=280,
    paretic_side=Side.RIGHT,
    session_type=Session.TEN_METER_BARE_CUE_OFF,
    distance=23.2
)
d = dataclasses.asdict(bag)

# p = "bag_info.json"
# with open(p, 'w', encoding='utf-8') as f:
#     json.dump(d, f, indent='\t')

from_dict(d)