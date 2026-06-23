from __future__ import annotations


class BtcBotError(Exception):
    pass


class ConfigError(BtcBotError):
    pass


class DataError(BtcBotError):
    pass


class ExchangeError(BtcBotError):
    pass


class GateBlocked(BtcBotError):
    pass


class StoreError(BtcBotError):
    pass


class LiveDisabledError(BtcBotError):
    pass
