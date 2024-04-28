from dataclasses import dataclass, fields
from enum import Enum


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

    def is_2MWT(self):
        trial_type = self.value.split('_')[0]
        return trial_type == '2MWT'



