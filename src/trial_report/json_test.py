import json
import dataclasses
from dataclasses import dataclass
from enum import Enum


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




bag_path = 'log_2023-08-16-15-40-08.bag'
bag_path2 = 'log_2023-08-16-15-41-19.bag'
test_date = bag_path.split('_')[1].split('-')
test_date = test_date[0]+test_date[1]+test_date[2]
test_label = 'RH-'+test_date[2:4]+'-01'
paretic_side = Side.LEFT
session_type = Session.TWO_MIN_ON_CUE_ON
ignore_cycle = (1, 2)
ignore_cycle2 = (4, 3)
bag1 = Bag(bag_path, test_date, test_label, paretic_side, session_type,ignore_cycle)
bag2 = Bag(bag_path2, test_date, test_label, paretic_side, session_type,ignore_cycle2)
bag1_dict = dataclasses.asdict(bag1)
bag2_dict = dataclasses.asdict(bag2)
d = [bag1_dict, bag2_dict]

file = "./test.json"
with open(file, 'r') as f:
    data = json.load(f)

print(load_bag(bag_path, data))

with open(file, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent='\t')

