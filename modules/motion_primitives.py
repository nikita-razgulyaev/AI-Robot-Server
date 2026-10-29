"""Motion primitives — кинематика локомоции.

Как задавать углы
=================
1. Карта каналов (индекс = номер в массиве из 22 углов):

   0  L_WING_FLAP     1  L_WING_FOLD     2  L_WING_TILT
   3  R_WING_FLAP     4  R_WING_FOLD     5  R_WING_TILT
   6  L_LEG_TURN      7  L_LEG_HIP       8  L_LEG_SHIN      9  L_LEG_TARSUS
   10 R_LEG_TURN      11 R_LEG_HIP       12 R_LEG_SHIN      13 R_LEG_TARSUS
   14 TAIL_YAW        15 TAIL_SPREAD     16 TAIL_LIFT
   17 HEAD_YAW        18 HEAD_PITCH      19 BEAK
   20 PELVIS_YAW      21 PELVIS_PITCH   (имена 17–21 уточните под свою механику)

2. Цикличные движения (шаг, взмах) — список SINGLE_* ниже.
   Каждый элемент: {"time": мс_от_начала_цикла, "servos": [22 угла]}.
   Первый и последний кадр цикла должны совпадать с нейтралью (или друг с другом),
   иначе на стыке повторов будет рывок.

   Удобный способ править отдельные каналы без переписывания всех 22:
       pose = patch(_NEUTRAL, {2: 60, 5: 120})   # CH02=60°, CH05=120°

3. Наклон хвоста / головы / таза — те же patch() в SINGLE_TURN_TICK / TAKEOFF / LANDING
   или в отдельных анимациях character/animations/*.json.

4. Физические константы STEP_LENGTH_CM / FLAP_ADVANCE_CM / TURN_STEP_DEG —
   сколько см/градусов даёт ОДИН повтор примитива. Замерьте на роботе и поправьте.

Как использовать:
    frames = build_walk(distance_cm=40)
    await servos.play_motion(frames, on_frame=...)
"""
from typing import Dict, List, Mapping, Optional, Sequence, Union

# === TUNE: физические константы примитивов ===
STEP_LENGTH_CM = 5.0
FLAP_ADVANCE_CM = 15.0
TURN_STEP_DEG = 15.0

# Имена каналов (для документации и patch по имени)
CHANNEL_NAMES = {
    0: "L_WING_FLAP", 1: "L_WING_FOLD", 2: "L_WING_TILT",
    3: "R_WING_FLAP", 4: "R_WING_FOLD", 5: "R_WING_TILT",
    6: "L_LEG_TURN", 7: "L_LEG_HIP", 8: "L_LEG_SHIN", 9: "L_LEG_TARSUS",
    10: "R_LEG_TURN", 11: "R_LEG_HIP", 12: "R_LEG_SHIN", 13: "R_LEG_TARSUS",
    14: "TAIL_YAW", 15: "TAIL_SPREAD", 16: "TAIL_LIFT",
    17: "HEAD_YAW", 18: "HEAD_PITCH", 19: "BEAK",
    20: "PELVIS_YAW", 21: "PELVIS_PITCH",
}
NAME_TO_CH = {v: k for k, v in CHANNEL_NAMES.items()}

_NEUTRAL: List[int] = [90] * 22


def patch(
    base_or_overrides: Optional[Union[Sequence[int], Mapping[Union[int, str], int]]] = None,
    overrides: Optional[Mapping[Union[int, str], int]] = None,
) -> List[int]:
    """Копия нейтрали (или base) с точечными правками каналов.

    Вызовы:
        patch({2: 60, 5: 120})
        patch({"TAIL_YAW": 110})
        patch(current_angles, {7: 80})
        patch()  # чистая нейтраль
    """
    base: Optional[Sequence[int]] = None
    ov = overrides
    if isinstance(base_or_overrides, Mapping):
        ov = base_or_overrides
    elif base_or_overrides is not None:
        base = base_or_overrides

    angles = list(base if base is not None else _NEUTRAL)
    if not ov:
        return angles
    for key, value in ov.items():
        if isinstance(key, int):
            idx = key
        else:
            s = str(key).strip().upper()
            if s in NAME_TO_CH:
                idx = NAME_TO_CH[s]
            elif s.startswith("CH"):
                idx = int(s[2:])
            else:
                raise KeyError(f"Неизвестный канал: {key!r}")
        if not 0 <= idx < 22:
            raise IndexError(f"Канал вне 0..21: {idx}")
        angles[idx] = int(value)
    return angles


# === TUNE: один цикл шага (заглушка по данным пользователя) ===
# Пользователь задал цикл «правая нога → левая нога» через CH02 / CH05.
# В карте README CH02 = L_WING_TILT, CH05 = R_WING_TILT — если на железе
# эти каналы другие, просто поменяйте номера в patch() ниже.
# Времена (мс) — заглушки; подгоните под реальную скорость шага.
#
# Правая:  CH02:60 → CH05:120 → CH02:120
# Левая:   CH02:60 → CH02:90  → CH02:60
SINGLE_STEP: List[Dict] = [
    {"time": 0,   "servos": patch()},                                      # нейтраль
    # --- правая нога ---
    {"time": 150, "servos": patch({2: 60})},                               # CH02: 60
    {"time": 300, "servos": patch({2: 60, 5: 120})},                       # CH05: 120
    {"time": 450, "servos": patch({2: 120, 5: 120})},                      # CH02: 120
    # --- левая нога ---
    {"time": 600, "servos": patch({2: 60})},                               # CH02: 60
    {"time": 750, "servos": patch({2: 90})},                               # CH02: 90
    {"time": 900, "servos": patch({2: 60})},                               # CH02: 60
    {"time": 1000, "servos": patch()},                                     # обратно в нейтраль
]

# === TUNE: один взмах в полёте ===
SINGLE_FLAP: List[Dict] = [
    {"time": 0,   "servos": patch()},
    {"time": 200, "servos": patch({0: 50, 3: 50})},    # крылья вверх (L/R_WING_FLAP)
    {"time": 400, "servos": patch({0: 130, 3: 130})},  # крылья вниз (рабочий ход)
    {"time": 600, "servos": patch()},
]

# === TUNE: один тик поворота (хвост) ===
# TURN_STEP_DEG ≈ насколько градусов корпуса даёт этот тик.
SINGLE_TURN_TICK: List[Dict] = [
    {"time": 0,   "servos": patch()},
    {"time": 200, "servos": patch({"TAIL_YAW": 110})},  # хвост в сторону (заглушка)
    {"time": 400, "servos": patch()},
]

# === TUNE: взлёт / посадка (одноразовые, не циклы) ===
TAKEOFF: List[Dict] = [
    {"time": 0,   "servos": patch()},
    {"time": 150, "servos": patch({7: 70, 11: 70})},           # присед (L/R_LEG_HIP)
    {"time": 300, "servos": patch({7: 110, 11: 110, 0: 50, 3: 50})},  # толчок + старт взмахов
    {"time": 450, "servos": patch({7: 90, 11: 90, 0: 130, 3: 130})},  # лапы поджаты, взмах
    {"time": 600, "servos": patch({0: 50, 3: 50})},
    {"time": 750, "servos": patch()},
]

LANDING: List[Dict] = [
    {"time": 0,   "servos": patch({"TAIL_LIFT": 70, 0: 100, 3: 100})},  # тормоз крыльями/хвостом
    {"time": 250, "servos": patch({7: 70, 11: 70, 8: 70, 12: 70})},     # лапы вперёд
    {"time": 500, "servos": patch({7: 90, 11: 90})},                    # касание
    {"time": 700, "servos": patch()},                                  # стойка
]


def _repeat_frames(single_cycle: List[Dict], count: int) -> List[Dict]:
    """Склеивает N повторов цикла со сквозным time. count <= 0 → []."""
    if count <= 0:
        return []
    frames: List[Dict] = []
    cycle_duration = single_cycle[-1]["time"]
    offset = 0
    for rep in range(count):
        for i, kf in enumerate(single_cycle):
            if rep > 0 and i == 0:
                continue  # не дублировать стыковой кадр
            frames.append({"time": offset + kf["time"], "servos": list(kf["servos"])})
        offset += cycle_duration
    return frames


def build_walk(distance_cm: float) -> List[Dict]:
    steps = round(abs(distance_cm) / STEP_LENGTH_CM)
    return _repeat_frames(SINGLE_STEP, steps)


def build_flight_forward(distance_cm: float) -> List[Dict]:
    flaps = round(abs(distance_cm) / FLAP_ADVANCE_CM)
    return _repeat_frames(SINGLE_FLAP, flaps)


def build_turn(angle_deg: float) -> List[Dict]:
    ticks = round(abs(angle_deg) / TURN_STEP_DEG)
    frames = _repeat_frames(SINGLE_TURN_TICK, ticks)
    # Зеркало для отрицательного угла: инвертируем TAIL_YAW относительно 90°
    if angle_deg < 0 and frames:
        for kf in frames:
            yaw = kf["servos"][14]
            kf["servos"][14] = 180 - yaw
    return frames


def build_takeoff() -> List[Dict]:
    return [{"time": kf["time"], "servos": list(kf["servos"])} for kf in TAKEOFF]


def build_landing() -> List[Dict]:
    return [{"time": kf["time"], "servos": list(kf["servos"])} for kf in LANDING]


def walk_duration_s(distance_cm: float) -> float:
    steps = round(abs(distance_cm) / STEP_LENGTH_CM)
    return 0.0 if steps <= 0 else steps * (SINGLE_STEP[-1]["time"] / 1000.0)


def turn_duration_s(angle_deg: float) -> float:
    ticks = round(abs(angle_deg) / TURN_STEP_DEG)
    return 0.0 if ticks <= 0 else ticks * (SINGLE_TURN_TICK[-1]["time"] / 1000.0)


def takeoff_duration_s() -> float:
    return TAKEOFF[-1]["time"] / 1000.0


def landing_duration_s() -> float:
    return LANDING[-1]["time"] / 1000.0


def flap_duration_s(distance_cm: float) -> float:
    flaps = round(abs(distance_cm) / FLAP_ADVANCE_CM)
    return 0.0 if flaps <= 0 else flaps * (SINGLE_FLAP[-1]["time"] / 1000.0)
