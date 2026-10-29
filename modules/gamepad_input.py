"""Геймпад — пункт 11 архитектуры.

Физически подключается к ПК по Bluetooth, читается через pygame.joystick.
Раскладка под DualShock 4 (SDL2 / pygame 2.x). Левый стик — ходьба/поворот
(вперёд/назад = дистанция, влево/вправо = поворот), правый стик и остальные
кнопки пока не назначены, ✕/○ — взлёт/посадка, L2/R2 — аналоговые триггеры (0…1).

Команды идут через RobotBrain.handle_command() с source="gamepad".
Локомоция активна только в control_mode == "manual".

on_state(callback) — для визуализации на панели: текущие оси и кнопки.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, Optional

from modules import motion_primitives
from modules.control_mode import control_mode

logger = logging.getLogger(__name__)

POLL_HZ = 30

# TUNE: индексы осей/кнопок под DualShock 4 (python -m pygame.examples.joystick)
AXIS_LX = 0
AXIS_LY = 1
AXIS_RX = 2
AXIS_RY = 3
AXIS_L2 = 4
AXIS_R2 = 5

BUTTON_CROSS = 0
BUTTON_CIRCLE = 1
BUTTON_SQUARE = 2
BUTTON_TRIANGLE = 3
BUTTON_L1 = 9
BUTTON_R1 = 10

BUTTON_LABELS = {
    BUTTON_CROSS: ("CROSS", "Взлёт"),
    BUTTON_CIRCLE: ("CIRCLE", "Посадка"),
    BUTTON_SQUARE: ("SQUARE", "—"),
    BUTTON_TRIANGLE: ("TRIANGLE", "—"),
    BUTTON_L1: ("L1", "—"),
    BUTTON_R1: ("R1", "—"),
}
TRIGGER_LABELS = {
    "l2": ("L2", "—"),
    "r2": ("R2", "—"),
}

DEADZONE = 0.15
MAX_STEP_DISTANCE_CM = 20.0
MAX_TURN_DEG = 30.0
MIN_COOLDOWN_S = 0.15
STATE_BROADCAST_HZ = 10


class GamepadInput:
    def __init__(self, handle_command, on_state: Optional[Callable[[Dict[str, Any]], Any]] = None):
        self._handle_command = handle_command
        self._on_state = on_state
        self._task: Optional[asyncio.Task] = None
        self._joystick = None
        self._cooldown_until = 0.0
        self._connected = False
        self._last_state_ts = 0.0
        self._name = ""
        self._trigger_rest = {AXIS_L2: -1.0, AXIS_R2: -1.0}

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self):
        if self._task is not None:
            return
        try:
            import pygame
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() == 0:
                logger.warning("[gamepad] Геймпад не найден — модуль не активируется")
                return
            self._joystick = pygame.joystick.Joystick(0)
            self._joystick.init()
            self._name = self._joystick.get_name()
            self._connected = True
            self._calibrate_triggers()
            logger.info(f"[gamepad] Подключен: {self._name}")
        except ImportError:
            logger.warning("[gamepad] pygame не установлен (pip install pygame)")
            return
        except Exception as e:
            logger.error(f"[gamepad] Ошибка инициализации: {e}")
            return

        self._task = asyncio.create_task(self._loop())

    def stop(self):
        if self._task is not None:
            self._task.cancel()
            self._task = None
        self._connected = False

    def _calibrate_triggers(self):
        j = self._joystick
        for axis in self._trigger_rest:
            raw = j.get_axis(axis) if j.get_numaxes() > axis else -1.0
            self._trigger_rest[axis] = raw if -1.01 <= raw <= 0.1 else -1.0

    def _trigger_value(self, axis: int) -> float:
        j = self._joystick
        if j.get_numaxes() <= axis:
            return 0.0
        rest = self._trigger_rest[axis]
        value = (j.get_axis(axis) - rest) / (1.0 - rest)
        return max(0.0, min(1.0, value))

    def _read_raw(self) -> Dict[str, Any]:
        j = self._joystick
        lx = j.get_axis(AXIS_LX) if j.get_numaxes() > AXIS_LX else 0.0
        ly = j.get_axis(AXIS_LY) if j.get_numaxes() > AXIS_LY else 0.0
        rx = j.get_axis(AXIS_RX) if j.get_numaxes() > AXIS_RX else 0.0
        ry = j.get_axis(AXIS_RY) if j.get_numaxes() > AXIS_RY else 0.0

        buttons = {}
        for idx, (label, purpose) in BUTTON_LABELS.items():
            pressed = bool(j.get_button(idx)) if j.get_numbuttons() > idx else False
            buttons[label] = {"pressed": pressed, "purpose": purpose, "index": idx}

        triggers = {}
        for key, axis in (("l2", AXIS_L2), ("r2", AXIS_R2)):
            label, purpose = TRIGGER_LABELS[key]
            triggers[key] = {"value": round(self._trigger_value(axis), 3), "label": label, "purpose": purpose}

        return {
            "type": "gamepad_state",
            "connected": True,
            "name": self._name,
            "control_mode": control_mode.mode,
            "axes": {
                "lx": {"value": round(lx, 3), "label": "LX", "purpose": "Поворот"},
                "ly": {"value": round(ly, 3), "label": "LY", "purpose": "Вперёд / назад"},
                "rx": {"value": round(rx, 3), "label": "RX", "purpose": "—"},
                "ry": {"value": round(ry, 3), "label": "RY", "purpose": "—"},
            },
            "buttons": buttons,
            "triggers": triggers,
        }

    async def _emit_state(self, state: Dict[str, Any]):
        if self._on_state is None:
            return
        try:
            result = self._on_state(state)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.debug(f"[gamepad] on_state error: {e}")

    async def _loop(self):
        import pygame

        try:
            while True:
                await asyncio.sleep(1.0 / POLL_HZ)
                pygame.event.pump()

                now = time.monotonic()
                state = self._read_raw()

                if now - self._last_state_ts >= 1.0 / STATE_BROADCAST_HZ:
                    self._last_state_ts = now
                    await self._emit_state(state)

                if not control_mode.is_manual():
                    continue

                forward = -state["axes"]["ly"]["value"]
                turn = state["axes"]["lx"]["value"]

                if abs(forward) > DEADZONE and now >= self._cooldown_until:
                    distance = forward * MAX_STEP_DISTANCE_CM
                    self._cooldown_until = now + max(
                        MIN_COOLDOWN_S, motion_primitives.walk_duration_s(distance)
                    )
                    asyncio.create_task(self._handle_command({
                        "type": "walk", "distance_cm": distance, "source": "gamepad",
                    }))
                elif abs(turn) > DEADZONE and now >= self._cooldown_until:
                    angle = turn * MAX_TURN_DEG
                    self._cooldown_until = now + max(
                        MIN_COOLDOWN_S, motion_primitives.turn_duration_s(angle)
                    )
                    asyncio.create_task(self._handle_command({
                        "type": "turn", "angle_deg": angle, "source": "gamepad",
                    }))

                btns = state["buttons"]
                if btns.get("CROSS", {}).get("pressed") and now >= self._cooldown_until:
                    self._cooldown_until = now + motion_primitives.takeoff_duration_s()
                    asyncio.create_task(self._handle_command({
                        "type": "takeoff", "source": "gamepad",
                    }))
                if btns.get("CIRCLE", {}).get("pressed") and now >= self._cooldown_until:
                    self._cooldown_until = now + motion_primitives.landing_duration_s()
                    asyncio.create_task(self._handle_command({
                        "type": "land", "source": "gamepad",
                    }))
        except asyncio.CancelledError:
            logger.info("[gamepad] Остановлен")
        except Exception as e:
            logger.error(f"[gamepad] Ошибка в цикле опроса: {e}")
            self._connected = False
