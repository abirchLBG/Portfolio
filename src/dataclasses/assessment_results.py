from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

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
        results: Nested dict mapping AssessmentName -> AssessmentType -> result value
        timer: Nested dict mapping AssessmentName -> AssessmentType -> elapsed time
    """

    results: dict["AssessmentName | str", dict[AssessmentType, float | pd.Series]] = (
        field(default_factory=dict)
    )
    timer: dict["AssessmentName | str", dict[AssessmentType, float]] = field(
        default_factory=dict
    )

    def get_summary_results(self) -> pd.DataFrame:
        """
        Extract all summary statistics into a DataFrame.

        Returns:
            DataFrame with assessments as index and summary values
        """
        summary_data = {}
        for assessment, types in self.results.items():
            if AssessmentType.Summary in types:
                summary_data[str(assessment)] = types[AssessmentType.Summary]

        return pd.DataFrame.from_dict(summary_data, orient="index", columns=["Value"])

    def get_rolling_results(self) -> pd.DataFrame:
        """
        Extract all rolling statistics into a DataFrame.

        Returns:
            DataFrame with datetime index and assessments as columns
        """
        rolling_data = {}
        for assessment, types in self.results.items():
            if AssessmentType.Rolling in types:
                rolling_data[str(assessment)] = types[AssessmentType.Rolling]

        return pd.DataFrame(rolling_data)

    def get_expanding_results(self) -> pd.DataFrame:
        """
        Extract all expanding statistics into a DataFrame.

        Returns:
            DataFrame with datetime index and assessments as columns
        """
        expanding_data = {}
        for assessment, types in self.results.items():
            if AssessmentType.Expanding in types:
                expanding_data[str(assessment)] = types[AssessmentType.Expanding]

        return pd.DataFrame(expanding_data)

    def get_result(
        self, assessment: "AssessmentName | str", assessment_type: AssessmentType
    ) -> float | pd.Series | None:
        """
        Get a specific result by assessment and type.

        Args:
            assessment: The assessment name
            assessment_type: The type of assessment (summary, rolling, expanding)

        Returns:
            The result value or None if not found
        """
        return self.results.get(assessment, {}).get(assessment_type)

    def timer_dataframe(self) -> pd.DataFrame:
        """
        Returns a DataFrame with timing stats.

        Returns:
            DataFrame with columns: Assessment, Type, Time (s)
        """
        rows = []
        for assessment, types in self.timer.items():
            for assessment_type, elapsed in types.items():
                rows.append(
                    {
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

        # Calculate total time per assessment
        assessment_totals = {}
        for assessment, types in self.timer.items():
            assessment_name = str(assessment)
            assessment_totals[assessment_name] = sum(types.values())

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

        # Count assessments and types
        num_assessments = len(assessment_totals)
        num_assessment_types = sum(len(types) for types in self.timer.values())

        # Calculate stats
        times = list(assessment_totals.values())
        mean_time = total_time / len(times)
        min_time = min(times)
        max_time = max(times)

        lines.append(f"Total Base Assessments | {num_assessments}")
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
        for assessment, types in self.timer.items():
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
        Create a heatmap visualization of results.

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
        for assessment, types in self.results.items():
            if assessment_type in types:
                value = types[assessment_type]
                if isinstance(value, (int, float)):
                    data[str(assessment)] = value

        if not data:
            raise ValueError(f"No {assessment_type} results available to plot")

        # Convert to DataFrame for heatmap
        df = pd.DataFrame.from_dict(data, orient="index", columns=["Value"])

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
        figsize: tuple[int, int] = (14, 6),
        title: str | None = None,
        return_fig: bool = False,
    ) -> Figure | None:
        """
        Create a comparison plot showing all three assessment types for a single assessment.

        Args:
            assessment: The assessment to compare
            figsize: Figure size as (width, height)
            title: Plot title (auto-generated if None)

        Returns:
            matplotlib Figure object
        """
        if assessment not in self.results:
            raise ValueError(f"Assessment '{assessment}' not found in results")

        assessment_results = self.results[assessment]

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
        num_assessments = len(self.results)
        num_types = sum(len(types) for types in self.results.values())
        return f"EvaluationResults(assessments={num_assessments}, total_results={num_types})"
