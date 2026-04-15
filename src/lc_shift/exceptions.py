from __future__ import annotations


class LCShiftError(Exception):
    pass


class ConfigurationError(LCShiftError):
    pass


class RoutingError(LCShiftError):
    pass


class BudgetExhaustedError(RoutingError):
    pass
