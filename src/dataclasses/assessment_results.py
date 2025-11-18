from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from src.dataclasses.assessment_config import AssessmentConfig

if TYPE_CHECKING:
    from src.constants import AssessmentName


class AssessmentType(StrEnum):
    Summary = "summary"
    Rolling = "rolling"
    Expanding = "expanding"


@dataclass
class EvaluationResults:
    """
    Container for all evaluation results with visualization capabilities.

    Attributes:
        results: Nested dict mapping config_key -> AssessmentName -> AssessmentType -> result value
        timer: Nested dict mapping config_key -> AssessmentName -> AssessmentType -> elapsed time
        config: The AssessmentConfig used to generate these results
        results_dfs: Dict mapping AssessmentType -> DataFrame with multi-level columns
    """

    results: dict[
        str, dict["AssessmentName | str", dict[AssessmentType, float | pd.Series]]
    ] = field(default_factory=dict)
    timer: dict[str, dict["AssessmentName | str", dict[AssessmentType, float]]] = field(
        default_factory=dict
    )
    config: AssessmentConfig | None = None
    results_dfs: dict[AssessmentType, pd.DataFrame] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        """Build results_dfs automatically after initialization."""
        self.build_results_dfs()

    def build_results_dfs(self) -> None:
        """
        Build structured DataFrames organized by assessment type with multi-level columns.

        For Summary: Index is assessments, columns are (Portfolio, RFR, Benchmark)
        For Rolling/Expanding: Index is dates, columns are (Portfolio, RFR, Benchmark, Assessment)
        """
        # Build Summary DataFrame
        summary_data = {}
        for config_key, config_results in self.results.items():
            # Parse config key: "portfolio|rfr|bmk"
            parts = config_key.split("|")
            if len(parts) != 3:
                continue

            portfolio, rfr, bmk = parts

            for assessment, types in config_results.items():
                if AssessmentType.Summary in types:
                    if assessment not in summary_data:
                        summary_data[assessment] = {}
                    summary_data[assessment][(portfolio, rfr, bmk)] = types[
                        AssessmentType.Summary
                    ]

        if summary_data:
            summary_df = pd.DataFrame(summary_data).T
            summary_df.columns = pd.MultiIndex.from_tuples(
                summary_df.columns, names=["Portfolio", "RFR", "Benchmark"]
            )
            self.results_dfs[AssessmentType.Summary] = summary_df

        # Build Rolling DataFrame
        rolling_data = {}
        for config_key, config_results in self.results.items():
            parts = config_key.split("|")
            if len(parts) != 3:
                continue

            portfolio, rfr, bmk = parts

            for assessment, types in config_results.items():
                if AssessmentType.Rolling in types:
                    col_key = (portfolio, rfr, bmk, str(assessment))
                    rolling_data[col_key] = types[AssessmentType.Rolling]

        if rolling_data:
            rolling_df = pd.DataFrame(rolling_data)
            rolling_df.columns = pd.MultiIndex.from_tuples(
                rolling_df.columns,
                names=["Portfolio", "RFR", "Benchmark", "Assessment"],
            )
            self.results_dfs[AssessmentType.Rolling] = rolling_df

        # Build Expanding DataFrame
        expanding_data = {}
        for config_key, config_results in self.results.items():
            parts = config_key.split("|")
            if len(parts) != 3:
                continue

            portfolio, rfr, bmk = parts

            for assessment, types in config_results.items():
                if AssessmentType.Expanding in types:
                    col_key = (portfolio, rfr, bmk, str(assessment))
                    expanding_data[col_key] = types[AssessmentType.Expanding]

        if expanding_data:
            expanding_df = pd.DataFrame(expanding_data)
            expanding_df.columns = pd.MultiIndex.from_tuples(
                expanding_df.columns,
                names=["Portfolio", "RFR", "Benchmark", "Assessment"],
            )
            self.results_dfs[AssessmentType.Expanding] = expanding_df

    def get_summary_results(self) -> pd.DataFrame:
        """
        Extract all summary statistics into a DataFrame with multilevel columns.

        Returns:
            DataFrame with assessments as index and config combinations as columns
        """
        summary_data = {}
        for config_key, config_results in self.results.items():
            for assessment, types in config_results.items():
                if AssessmentType.Summary in types:
                    if assessment not in summary_data:
                        summary_data[assessment] = {}
                    summary_data[assessment][config_key] = types[AssessmentType.Summary]

        # Convert to DataFrame
        df = pd.DataFrame(summary_data).T

        # Parse config keys into multilevel columns if multiple configs
        if len(df.columns) > 1 and "|" in str(df.columns[0]):
            # Split config keys into (returns, rfr, bmk)
            new_cols = []
            for col in df.columns:
                parts = col.split("|")
                if len(parts) == 3:
                    new_cols.append(tuple(parts))
                else:
                    new_cols.append(col)

            if all(isinstance(c, tuple) for c in new_cols):
                df.columns = pd.MultiIndex.from_tuples(
                    new_cols, names=["Returns", "RFR", "Benchmark"]
                )

        return df

    def get_rolling_results(self) -> pd.DataFrame:
        """
        Extract all rolling statistics into a DataFrame with multilevel columns.

        Returns:
            DataFrame with datetime index and multilevel columns (config, assessment)
        """
        rolling_data = {}
        for config_key, config_results in self.results.items():
            for assessment, types in config_results.items():
                if AssessmentType.Rolling in types:
                    # Create multilevel column key
                    col_key = (config_key, str(assessment))
                    rolling_data[col_key] = types[AssessmentType.Rolling]

        df = pd.DataFrame(rolling_data)

        # Create multilevel columns
        if df.columns.size > 0 and isinstance(df.columns[0], tuple):
            # Parse config keys if they contain pipe separators
            new_cols = []
            for config_key, assessment in df.columns:
                parts = config_key.split("|")
                if len(parts) == 3:
                    new_cols.append((*parts, assessment))
                else:
                    new_cols.append((config_key, assessment))

            if all(len(c) == 4 for c in new_cols):
                df.columns = pd.MultiIndex.from_tuples(
                    new_cols, names=["Returns", "RFR", "Benchmark", "Assessment"]
                )
            else:
                df.columns = pd.MultiIndex.from_tuples(
                    df.columns, names=["Config", "Assessment"]
                )

        return df

    def get_expanding_results(self) -> pd.DataFrame:
        """
        Extract all expanding statistics into a DataFrame with multilevel columns.

        Returns:
            DataFrame with datetime index and multilevel columns (config, assessment)
        """
        expanding_data = {}
        for config_key, config_results in self.results.items():
            for assessment, types in config_results.items():
                if AssessmentType.Expanding in types:
                    # Create multilevel column key
                    col_key = (config_key, str(assessment))
                    expanding_data[col_key] = types[AssessmentType.Expanding]

        df = pd.DataFrame(expanding_data)

        # Create multilevel columns
        if df.columns.size > 0 and isinstance(df.columns[0], tuple):
            # Parse config keys if they contain pipe separators
            new_cols = []
            for config_key, assessment in df.columns:
                parts = config_key.split("|")
                if len(parts) == 3:
                    new_cols.append((*parts, assessment))
                else:
                    new_cols.append((config_key, assessment))

            if all(len(c) == 4 for c in new_cols):
                df.columns = pd.MultiIndex.from_tuples(
                    new_cols, names=["Returns", "RFR", "Benchmark", "Assessment"]
                )
            else:
                df.columns = pd.MultiIndex.from_tuples(
                    df.columns, names=["Config", "Assessment"]
                )

        return df

    def get_result(
        self,
        assessment: "AssessmentName | str",
        assessment_type: AssessmentType,
        config_key: str | None = None,
    ) -> float | pd.Series | dict | None:
        """
        Get a specific result by assessment and type.

        Args:
            assessment: The assessment name
            assessment_type: The type of assessment (summary, rolling, expanding)
            config_key: Optional config key to get result for specific config.
                       If None, returns dict of all configs.

        Returns:
            The result value, dict of results, or None if not found
        """
        if config_key is not None:
            return (
                self.results.get(config_key, {})
                .get(assessment, {})
                .get(assessment_type)
            )

        # Return all configs for this assessment/type
        results = {}
        for cfg_key, config_results in self.results.items():
            if (
                assessment in config_results
                and assessment_type in config_results[assessment]
            ):
                results[cfg_key] = config_results[assessment][assessment_type]

        return results if results else None

    def timer_dataframe(self) -> pd.DataFrame:
        """
        Returns a DataFrame with timing stats.

        Returns:
            DataFrame with columns: Config, Assessment, Type, Time (s)
        """
        rows = []
        for config_key, config_timer in self.timer.items():
            for assessment, types in config_timer.items():
                for assessment_type, elapsed in types.items():
                    rows.append(
                        {
                            "Config": config_key,
                            "Assessment": str(assessment),
                            "Type": assessment_type,
                            "Time (s)": elapsed,
                        }
                    )
        return pd.DataFrame(rows)

    def timer_report(self, max_bar_width: int = 50) -> str:
        """
        Generate a formatted timing report with visual bars and summary statistics.

        Args:
            max_bar_width: Maximum width of the timing bars in characters

        Returns:
            Formatted string report with timing information and bars
        """
        if not self.timer:
            return "No timing data available. Run evaluation first."

        # Calculate total time per assessment across all configs
        assessment_totals = {}
        for config_key, config_timer in self.timer.items():
            for assessment, types in config_timer.items():
                assessment_name = str(assessment)
                if assessment_name not in assessment_totals:
                    assessment_totals[assessment_name] = 0
                assessment_totals[assessment_name] += sum(types.values())

        # Sort by total time (descending)
        sorted_assessments = sorted(
            assessment_totals.items(), key=lambda x: x[1], reverse=True
        )

        # Calculate total time and percentages
        total_time = sum(assessment_totals.values())

        if total_time == 0:
            return "No timing data available (total time is 0)."

        # Build report header
        n_chars = max_bar_width + 42
        lines = []
        lines.append("=" * n_chars)
        lines.append("ASSESSMENT TIMING REPORT")
        lines.append("=" * n_chars)
        lines.append("")

        # Build timing bars
        for assessment_name, elapsed in sorted_assessments:
            percentage = (elapsed / total_time) * 100
            bar_length = int((elapsed / total_time) * max_bar_width)
            bar = "█" * bar_length

            # Format the line with assessment name, bar, time, and percentage
            lines.append(
                f"{assessment_name:<20}|{bar:<{max_bar_width}}|{percentage:>5.1f}% | {elapsed:>8.3f}s |"
            )

        # Add summary statistics
        lines.append("")
        lines.append("-" * n_chars)
        lines.append("SUMMARY STATISTICS")
        lines.append("-" * n_chars)

        # Count assessments, types, and configs
        num_assessments = len(assessment_totals)
        num_configs = len(self.timer)
        num_assessment_types = 0
        for config_timer in self.timer.values():
            num_assessment_types += sum(len(types) for types in config_timer.values())

        # Calculate stats
        times = list(assessment_totals.values())
        mean_time = total_time / len(times)
        min_time = min(times)
        max_time = max(times)

        lines.append(f"Total Base Assessments | {num_assessments}")
        lines.append(f"    Total Configurations | {num_configs}")
        lines.append(f" Total Assessment Runs | {num_assessment_types}")
        lines.append(f"            Total Time | {total_time:.3f}s")
        lines.append(f"             Mean Time | {mean_time:.3f}s")
        lines.append(
            f"              Max Time | {max_time:.3f}s ({max(sorted_assessments, key=lambda x: x[1])[0]})"
        )
        lines.append(
            f"              Min Time | {min_time:.3f}s ({min(sorted_assessments, key=lambda x: x[1])[0]})"
        )

        # Show breakdown by assessment type if we have multiple types
        type_totals = {}
        for config_timer in self.timer.values():
            for assessment, types in config_timer.items():
                for assessment_type, elapsed in types.items():
                    type_name = (
                        assessment_type.value
                        if hasattr(assessment_type, "value")
                        else str(assessment_type)
                    )
                    type_totals[type_name] = type_totals.get(type_name, 0) + elapsed

        if len(type_totals) > 1:
            lines.append("")
            lines.append("Time by Assessment Type:")
            for type_name, elapsed in sorted(
                type_totals.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = (elapsed / total_time) * 100
                lines.append(
                    f"  {type_name:<12} {percentage:>5.1f}%  | {elapsed:>7.3f}s"
                )

        lines.append("=" * n_chars)

        final = "\n".join(lines)
        print(final)
        return final

    def plot_summary(
        self,
        figsize: tuple[int, int] = (12, 6),
        title: str = "Assessment Summary Results",
        return_fig: bool = False,
    ) -> Figure | None:
        """
        Create a bar plot of summary statistics.

        Args:
            figsize: Figure size as (width, height)
            title: Plot title

        Returns:
            matplotlib Figure object
        """
        df = self.get_summary_results()

        if df.empty:
            raise ValueError("No summary results available to plot")

        fig, ax = plt.subplots(figsize=figsize)
        _ = df.plot(kind="bar", ax=ax, legend=False)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Assessment", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()

        if return_fig:
            return fig

    def plot_rolling(
        self,
        assessments: list["AssessmentName | str"] | None = None,
        figsize: tuple[int, int] = (14, 8),
        title: str = "Rolling Assessment Results",
        return_fig: bool = False,
    ) -> Figure | None:
        """
        Create a line plot of rolling statistics over time.

        Args:
            assessments: List of specific assessments to plot (None = all)
            figsize: Figure size as (width, height)
            title: Plot title

        Returns:
            matplotlib Figure object
        """
        df = self.get_rolling_results()

        if df.empty:
            raise ValueError("No rolling results available to plot")

        if assessments:
            available_cols = [str(a) for a in assessments if str(a) in df.columns]
            df = df[available_cols]

        fig, ax = plt.subplots(figsize=figsize)
        _ = df.plot(ax=ax)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.legend(title="Assessment", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        if return_fig:
            return fig

    def plot_expanding(
        self,
        assessments: list["AssessmentName | str"] | None = None,
        figsize: tuple[int, int] = (14, 8),
        title: str = "Expanding Assessment Results",
        return_fig: bool = False,
    ) -> Figure | None:
        """
        Create a line plot of expanding statistics over time.

        Args:
            assessments: List of specific assessments to plot (None = all)
            figsize: Figure size as (width, height)
            title: Plot title

        Returns:
            matplotlib Figure object
        """
        df = self.get_expanding_results()

        if df.empty:
            raise ValueError("No expanding results available to plot")

        if assessments:
            available_cols = [str(a) for a in assessments if str(a) in df.columns]
            df = df[available_cols]

        fig, ax = plt.subplots(figsize=figsize)
        _ = df.plot(ax=ax)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.legend(title="Assessment", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        if return_fig:
            return fig

    def plot_heatmap(
        self,
        assessment_type: AssessmentType = AssessmentType.Summary,
        figsize: tuple[int, int] = (10, 8),
        cmap: str = "RdYlGn",
        title: str | None = None,
        return_fig: bool = False,
    ) -> Figure | None:
        """
        Create a heatmap visualization of results across configs and assessments.

        Args:
            assessment_type: Which type of results to visualize
            figsize: Figure size as (width, height)
            cmap: Colormap to use
            title: Plot title (auto-generated if None)

        Returns:
            matplotlib Figure object
        """
        # Collect data for the specified type
        data = {}
        for config_key, config_results in self.results.items():
            for assessment, types in config_results.items():
                if assessment_type in types:
                    value = types[assessment_type]
                    if isinstance(value, (int, float)):
                        if assessment not in data:
                            data[assessment] = {}
                        data[assessment][config_key] = value

        if not data:
            raise ValueError(f"No {assessment_type} results available to plot")

        # Convert to DataFrame for heatmap (assessments x configs)
        df = pd.DataFrame(data).T

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            df, annot=True, fmt=".4f", cmap=cmap, ax=ax, cbar_kws={"label": "Value"}
        )

        if title is None:
            title = f"{assessment_type.value.capitalize()} Results Heatmap"
        ax.set_title(title, fontsize=14, fontweight="bold")
        fig.tight_layout()

        if return_fig:
            return fig

    def plot_comparison(
        self,
        assessment: "AssessmentName | str",
        config_key: str | None = None,
        figsize: tuple[int, int] = (14, 6),
        title: str | None = None,
        return_fig: bool = False,
    ) -> Figure | None:
        """
        Create a comparison plot showing all three assessment types for a single assessment.

        Args:
            assessment: The assessment to compare
            config_key: Optional config key. If None, uses first config.
            figsize: Figure size as (width, height)
            title: Plot title (auto-generated if None)

        Returns:
            matplotlib Figure object
        """
        # Get config key
        if config_key is None:
            if not self.results:
                raise ValueError("No results available")
            config_key = list(self.results.keys())[0]

        if config_key not in self.results:
            raise ValueError(f"Config '{config_key}' not found in results")

        if assessment not in self.results[config_key]:
            raise ValueError(
                f"Assessment '{assessment}' not found in config '{config_key}'"
            )

        assessment_results = self.results[config_key][assessment]

        # Determine which types are available
        has_summary = AssessmentType.Summary in assessment_results
        has_rolling = AssessmentType.Rolling in assessment_results
        has_expanding = AssessmentType.Expanding in assessment_results

        # Count available plots
        num_plots = sum([has_rolling, has_expanding])

        if num_plots == 0 and has_summary:
            raise ValueError(
                f"Only summary data available for {assessment}. Use plot_summary() instead."
            )

        if num_plots == 0:
            raise ValueError(f"No plottable data available for {assessment}")

        # Create subplots
        fig, axes = plt.subplots(1, num_plots, figsize=figsize)
        if num_plots == 1:
            axes = [axes]

        if title is None:
            title = f"{assessment} - Assessment Type Comparison"
        fig.suptitle(title, fontsize=14, fontweight="bold")

        plot_idx = 0

        if has_rolling:
            rolling_data = assessment_results[AssessmentType.Rolling]
            if isinstance(rolling_data, pd.Series):
                axes[plot_idx].plot(rolling_data.index, rolling_data.values)
                axes[plot_idx].set_title("Rolling")
                axes[plot_idx].set_xlabel("Date")
                axes[plot_idx].set_ylabel("Value")
                axes[plot_idx].grid(alpha=0.3)
                plot_idx += 1

        if has_expanding:
            expanding_data = assessment_results[AssessmentType.Expanding]
            if isinstance(expanding_data, pd.Series):
                axes[plot_idx].plot(expanding_data.index, expanding_data.values)
                axes[plot_idx].set_title("Expanding")
                axes[plot_idx].set_xlabel("Date")
                axes[plot_idx].set_ylabel("Value")
                axes[plot_idx].grid(alpha=0.3)
                plot_idx += 1

        fig.tight_layout()

        if return_fig:
            return fig

    def plot_timing(
        self, figsize: tuple[int, int] = (12, 6), return_fig: bool = False
    ) -> Figure | None:
        """
        Create a visualization of timing data.

        Args:
            figsize: Figure size as (width, height)

        Returns:
            matplotlib Figure object
        """
        df = self.timer_dataframe()

        if df.empty:
            raise ValueError("No timing data available to plot")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Time by assessment
        assessment_totals = (
            df.groupby("Assessment")["Time (s)"].sum().sort_values(ascending=False)
        )
        _ = assessment_totals.plot(kind="barh", ax=ax1)
        ax1.set_title("Total Time by Assessment", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Time (s)")
        ax1.grid(axis="x", alpha=0.3)

        # Plot 2: Time by assessment type
        type_totals = df.groupby("Type")["Time (s)"].sum().sort_values(ascending=False)
        _ = type_totals.plot(kind="bar", ax=ax2)
        ax2.set_title("Total Time by Assessment Type", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Time (s)")
        ax2.set_xlabel("Assessment Type")
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)
        ax2.grid(axis="y", alpha=0.3)

        fig.tight_layout()

        if return_fig:
            return fig

    def __repr__(self) -> str:
        num_configs = len(self.results)
        num_assessments = 0
        num_results = 0

        for config_results in self.results.values():
            num_assessments = max(num_assessments, len(config_results))
            num_results += sum(len(types) for types in config_results.values())

        lines = [
            "EvaluationResults(",
            f"  configurations={num_configs}",
            f"  unique_assessments={num_assessments}",
            f"  total_results={num_results}",
        ]

        if self.config:
            lines.append(f"  overlap_mode={self.config.overlap_mode.value}")

        lines.append(")")
        return "\n".join(lines)
