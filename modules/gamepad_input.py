"""Геймпад — пункт 11 архитектуры.

Физически подключается к ПК по Bluetooth, читается через pygame.joystick.
Левый стик — ходьба/полёт (вперёд/назад = дистанция, влево/вправо = поворот),
правый стик — зарезервирован (пока не используется), кнопки — взлёт/посадка.

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

# TUNE: индексы осей/кнопок — подберите под свой геймпад
# (python -m pygame.examples.joystick)
AXIS_LX = 0          # левый стик X  (−1 = влево, +1 = вправо) → поворот
AXIS_LY = 1          # левый стик Y  (−1 = вверх, +1 = вниз)  → вперёд/назад (инвертируем)
AXIS_RX = 2          # правый стик X (зарезервирован)
AXIS_RY = 3          # правый стик Y (зарезервирован)
AXIS_LT = 4          # L2 / LT (если аналоговый)
AXIS_RT = 5          # R2 / RT

BUTTON_A = 0         # A / Cross     → взлёт
BUTTON_B = 1         # B / Circle    → посадка
BUTTON_X = 2         # X / Square
BUTTON_Y = 3         # Y / Triangle
BUTTON_LB = 4        # L1 / LB
BUTTON_RB = 5        # R1 / RB
BUTTON_SELECT = 6
BUTTON_START = 7

# Подписи для панели (имя на схеме → назначение)
BUTTON_LABELS = {
    BUTTON_A: ("A", "Взлёт"),
    BUTTON_B: ("B", "Посадка"),
    BUTTON_X: ("X", "—"),
    BUTTON_Y: ("Y", "—"),
    BUTTON_LB: ("LB", "—"),
    BUTTON_RB: ("RB", "—"),
    BUTTON_SELECT: ("Select", "—"),
    BUTTON_START: ("Start", "—"),
}
AXIS_LABELS = {
    "lx": ("LX", "Поворот"),
    "ly": ("LY", "Вперёд / назад"),
    "rx": ("RX", "—"),
    "ry": ("RY", "—"),
}

DEADZONE = 0.15
MAX_STEP_DISTANCE_CM = 20.0
MAX_TURN_DEG = 30.0
MIN_COOLDOWN_S = 0.15
# Как часто слать gamepad_state на панель (не каждый poll — экономим WS)
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

                # Визуализация — чаще, чем команды
                if now - self._last_state_ts >= 1.0 / STATE_BROADCAST_HZ:
                    self._last_state_ts = now
                    await self._emit_state(state)

                # Локомоция только в manual
                if not control_mode.is_manual():
                    continue

                forward = -state["axes"]["ly"]["value"]  # вверх стика = вперёд
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
                if btns.get("A", {}).get("pressed") and now >= self._cooldown_until:
                    self._cooldown_until = now + motion_primitives.takeoff_duration_s()
                    asyncio.create_task(self._handle_command({
                        "type": "takeoff", "source": "gamepad",
                    }))
                if btns.get("B", {}).get("pressed") and now >= self._cooldown_until:
                    self._cooldown_until = now + motion_primitives.landing_duration_s()
                    asyncio.create_task(self._handle_command({
                        "type": "land", "source": "gamepad",
                    }))
        except asyncio.CancelledError:
            logger.info("[gamepad] Остановлен")
        except Exception as e:
            logger.error(f"[gamepad] Ошибка в цикле опроса: {e}")
            self._connected = False
