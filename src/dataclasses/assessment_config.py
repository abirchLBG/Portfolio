from abc import ABC
from dataclasses import asdict, dataclass, field
from datetime import date
from enum import StrEnum
from itertools import product
import logging
from typing import Iterator

import pandas as pd

logger = logging.getLogger(__name__)


class OverlapMode(StrEnum):
    """Mode for handling overlapping periods between multiple series."""

    FULL = "full"  # Use full track for each series (default)
    LONGEST_OVERLAP = "longest_overlap"  # Use only the longest overlapping period


@dataclass(kw_only=True)
class AssessmentConfig(ABC):
    returns: pd.Series | list[pd.Series]
    rfr: pd.Series | list[pd.Series]
    bmk: pd.Series | list[pd.Series]

    start: str | pd.Timestamp | date | None = None
    end: str | pd.Timestamp | date | None = None

    ann_factor: int = 252
    window: int = 252
    min_periods: int = 21  # 1 BMonth
    overlap_mode: OverlapMode = OverlapMode.FULL

    # Internal fields for normalized lists
    _returns_list: list[pd.Series] = field(init=False, repr=False)
    _rfr_list: list[pd.Series] = field(init=False, repr=False)
    _bmk_list: list[pd.Series] = field(init=False, repr=False)

    def __post_init__(self):
        # Validate numeric parameters BEFORE any data processing
        if self.ann_factor <= 0:
            raise ValueError(f"ann_factor must be positive, got {self.ann_factor}")

        if self.window <= 0:
            raise ValueError(f"window must be positive, got {self.window}")

        if self.min_periods <= 0:
            raise ValueError(f"min_periods must be positive, got {self.min_periods}")

        if self.min_periods > self.window:
            logger.warning(
                f"min_periods ({self.min_periods}) is greater than window ({self.window}). "
                "This may cause issues in rolling calculations."
            )

        # Normalize inputs to lists
        self._returns_list = (
            self.returns if isinstance(self.returns, list) else [self.returns]
        )
        self._rfr_list = self.rfr if isinstance(self.rfr, list) else [self.rfr]
        self._bmk_list = self.bmk if isinstance(self.bmk, list) else [self.bmk]

        # Validate all series are not empty
        for i, ret in enumerate(self._returns_list):
            if ret.empty:
                raise ValueError(f"returns series at index {i} cannot be empty")

        for i, rfr in enumerate(self._rfr_list):
            if rfr.empty:
                raise ValueError(f"rfr series at index {i} cannot be empty")

        for i, bmk in enumerate(self._bmk_list):
            if bmk.empty:
                raise ValueError(f"bmk series at index {i} cannot be empty")

        # Validate minimum data length for returns
        min_required_length = max(
            self.min_periods, 2
        )  # Need at least 2 points for calculations
        for i, ret in enumerate(self._returns_list):
            if len(ret) < min_required_length:
                raise ValueError(
                    f"returns at index {i} must have at least {min_required_length} data points, "
                    f"got {len(ret)}"
                )

        # Store kwargs for later use (excluding internal fields)
        # Note: For single-series configs, we keep the series in kwargs for serialization
        base_kwargs = {
            k: v
            for k, v in asdict(self).items()
            if not k.startswith("_") and k not in ["returns", "rfr", "bmk"]
        }

        # If single series (not lists), add them to kwargs for backward compatibility
        if (
            len(self._returns_list) == 1
            and len(self._rfr_list) == 1
            and len(self._bmk_list) == 1
        ):
            base_kwargs["returns"] = self._returns_list[0]
            base_kwargs["rfr"] = self._rfr_list[0]
            base_kwargs["bmk"] = self._bmk_list[0]

        self.kwargs: dict = base_kwargs

    def _process_single_config(
        self, returns: pd.Series, rfr: pd.Series, bmk: pd.Series
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Process a single combination of returns, rfr, and bmk.

        Returns:
            Tuple of (processed_returns, processed_rfr, processed_bmk)
        """
        # Apply date filtering
        if self.start is not None:
            start_date = pd.Timestamp(self.start).date()
            returns = returns[start_date:]

        if self.end is not None:
            end_date = pd.Timestamp(self.end).date()
            returns = returns[:end_date]

        # Handle overlap mode
        if self.overlap_mode == OverlapMode.LONGEST_OVERLAP:
            # Find the longest overlapping period
            all_series = [returns, rfr, bmk]
            start_idx = max(s.first_valid_index() for s in all_series if not s.empty)
            end_idx = min(s.last_valid_index() for s in all_series if not s.empty)

            if start_idx > end_idx:
                raise ValueError(
                    f"No overlapping period found for {returns.name}, {rfr.name}, {bmk.name}"
                )

            returns = returns[start_idx:end_idx]
            rfr = rfr[start_idx:end_idx]
            bmk = bmk[start_idx:end_idx]

        # Reindex rfr and bmk to match returns
        rfr = rfr.reindex(returns.index).fillna(0)
        bmk = bmk.reindex(returns.index).fillna(0)

        # Validate processed data
        if rfr.isna().any():
            raise ValueError(
                f"Risk-free rate ({rfr.name}) has missing data after processing"
            )

        if returns.isna().any():
            raise ValueError(
                f"Returns ({returns.name}) have missing data after processing"
            )

        if not returns.index.equals(rfr.index):
            raise ValueError(
                f"Index mismatch between returns ({returns.name}) and rfr ({rfr.name})"
            )

        return returns, rfr, bmk

    def iter_configs(self) -> Iterator[tuple[str, "AssessmentConfig"]]:
        """
        Generate individual AssessmentConfig instances for each combination.

        Yields:
            Tuple of (config_key, config) where config_key identifies the combination
        """
        for ret, rfr, bmk in product(
            self._returns_list, self._rfr_list, self._bmk_list
        ):
            # Process the combination
            processed_ret, processed_rfr, processed_bmk = self._process_single_config(
                ret, rfr, bmk
            )

            # Create a key for this combination
            config_key = (
                f"{processed_ret.name}|{processed_rfr.name}|{processed_bmk.name}"
            )

            # Create a new config instance for this combination
            # We create a new instance to avoid modifying the original
            config = SingleAssessmentConfig(
                returns=processed_ret,
                rfr=processed_rfr,
                bmk=processed_bmk,
                start=self.start,
                end=self.end,
                ann_factor=self.ann_factor,
                window=self.window,
                min_periods=self.min_periods,
                overlap_mode=self.overlap_mode,
            )

            yield config_key, config


@dataclass(kw_only=True)
class SingleAssessmentConfig(AssessmentConfig):
    """
    Internal class representing a single configuration.
    Used by assessments to process individual combinations.
    """

    returns: pd.Series
    rfr: pd.Series
    bmk: pd.Series

    def __post_init__(self):
        # For single configs, validation is simpler
        if self.ann_factor <= 0:
            raise ValueError(f"ann_factor must be positive, got {self.ann_factor}")

        if self.window <= 0:
            raise ValueError(f"window must be positive, got {self.window}")

        if self.min_periods <= 0:
            raise ValueError(f"min_periods must be positive, got {self.min_periods}")

        if self.min_periods > self.window:
            logger.warning(
                f"min_periods ({self.min_periods}) is greater than window ({self.window}). "
                "This may cause issues in rolling calculations."
            )

        # No processing needed - data is already processed
        # Just validate it's not empty
        if self.returns.empty:
            raise ValueError("returns series cannot be empty")

        if self.rfr.empty:
            raise ValueError("rfr series cannot be empty")

        if self.bmk.empty:
            raise ValueError("bmk series cannot be empty")

        # Initialize the list fields to avoid asdict errors
        self._returns_list = [self.returns]
        self._rfr_list = [self.rfr]
        self._bmk_list = [self.bmk]

        # Create kwargs manually to avoid issues with pd.Series serialization
        self.kwargs: dict = {
            "returns": self.returns,
            "rfr": self.rfr,
            "bmk": self.bmk,
            "start": self.start,
            "end": self.end,
            "ann_factor": self.ann_factor,
            "window": self.window,
            "min_periods": self.min_periods,
            "overlap_mode": self.overlap_mode,
        }
