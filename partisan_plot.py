from pathlib import Path
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PartisanPlot:
    default_draw_values = ['k', 200, '-', 0.2, 0.6, 11]

    def __init__(
            self,
            house_data_path: Path,
            num_districts_path: Path,
            years_path: Path,
            colors_path: Path):
        """Initializes variables used for drawing of partisan bias
        plots.

        Args:
            house_data_path (Path): Pathlib path to the MIT Election
                Lab U.S. House 1976–2018 CSV file
                (https://electionlab.mit.edu/data)
            num_districts_path (Path): Pathlib path to a CSV file
                containing the row-separated states that draw() will
                plot, and the number of congressional districts in each
                state.
            years_path (Path): Pathlib path to a CSV file containing
                row-separated years that each state will be drawn for.
            colors_path (Path): Pathlib path to a CSV file containing
                row-separated named colors in matplotlib, corresponding
                to each year in the file that years_path points to.
        """
        with open(house_data_path, 'r', errors='ignore') as \
                read_house_data:
            self.house_data = pd.read_csv(read_house_data)
        with open(num_districts_path, 'r', errors='ignore') as \
                read_states:
            self.num_districts = pd.read_csv(read_states, header=None,
                                             sep=' ', index_col=0,
                                             squeeze=True)
        with open(years_path, 'r', errors='ignore') as read_years:
            self.years = pd.read_csv(read_years, header=None,
                                     squeeze=True)
        with open(colors_path, 'r', errors='ignore') as read_colors:
            self.colors = pd.read_csv(read_colors, header=None,
                                      squeeze=True)
            self.colors.index = self.years

    def drop_columns(self, dropped_columns_path: Path):
        """Drops columns from self.house_data, as indicated in the CSV
        file that dropped_columns_path points to.

        Args:
            dropped_columns_path (Path): Pathlib path to a CSV file
                containing row-separated strings, representing
                indices of DataFrame columns that will be dropped.
        """
        with open(dropped_columns_path, 'r', errors='ignore') as \
                read_dropped_columns:
            dropped_columns = pd.read_csv(
                read_dropped_columns, header=None, squeeze=True)
            self.house_data.drop(dropped_columns, inplace=True, axis=1)

    def draw(self,
             state: str,
             figsize: Tuple[int,
                            int],
             axescolor: str = None,
             dotsize: int = None,
             axeslinestyle: str = None,
             axeslinewidth: float = None,
             stepalpha: float = None,
             num_yticks=None):
        """Draws partisan bias charts which are used in main.ipynb.

        Args:
            state: the postal abbreviation of the state to be drawn
            figsize: the figure size of the plot
            dotsize: the size of dot representing the actual vote margin
            axeslinestyle: the line style of plot axes
            axeslinewidth: the line style of plot axes
            stepalpha: the alpha (opacity) value of the plot "step"
                graph
            num_yticks: number of ticks denoting Democratic Share of
                Seats on the y-axis
        """

        # Because default arguments in Python are evaluated when the
        # function is defined, they are initialized as None in the
        # function definition. __draw_default_parameters assigns default
        # values, so that if any of the default arguments are mutated,
        # the argument is does not retain the mutated value in future
        # calls.

        [axescolor,
         dotsize,
         axeslinestyle,
         axeslinewidth,
         stepalpha,
         num_yticks] = self.__draw_default_parameters([axescolor,
                                                       dotsize,
                                                       axeslinestyle,
                                                       axeslinewidth,
                                                       stepalpha,
                                                       num_yticks])

        fig, axes = plt.subplots(figsize=figsize)
        axes.set_xlabel(
            'Average Democratic Vote Margin in US House Races '
            '(Two-Party Vote)')
        axes.set_ylabel('Democratic Share of Seats in the US House')

        # the values of the ticks on the y axis are spaced evenly based
        # on num_yticks
        yticks = np.linspace(0, 1, num_yticks)
        axes.set_yticks(yticks)
        axes.grid(
            color=axescolor,
            linestyle=axeslinestyle,
            linewidth=axeslinewidth)

        extrema = pd.Series((-1, 1), index=(0, self.num_districts))

        axes.plot([-1, 1], [0.5, 0.5], linewidth=3, color='k')
        axes.plot([0, 0], [0, 1], linewidth=3, color='k')

        for year in self.years:
            # only use the vote shares of Republican and Democratic
            # candidates, in case minor party or independent candidates
            # run in any of the studied elections
            major_party_candidates = self.house_data[
                (self.house_data['year'] == year) &
                (self.house_data['state_po'] == state) &
                ((self.house_data['party'] == 'republican') |
                 (self.house_data['party'] == 'democrat'))]
            vote_share = pd.DataFrame(
                np.zeros((self.num_districts[state], 2)),
                columns=['democrat', 'republican'])
            vote_share.index += 1
            dem_votes = rep_votes = 0

            # convert total vote for all candidates to Democratic vote
            # share, only accounting for two-party votes
            for index, candidate in major_party_candidates.iterrows():
                vote_share.loc[
                    candidate['district'], candidate['party']] += \
                    (candidate['candidatevotes'] / candidate[
                        'totalvotes'])
                if candidate['party'] == 'democrat':
                    dem_votes += candidate['candidatevotes']
                else:
                    rep_votes += candidate['candidatevotes']

            actual_margin = (dem_votes - rep_votes) / (
                dem_votes + rep_votes)
            extrapolated_margin = vote_share.loc[:, 'republican'] - \
                vote_share.loc[:, 'democrat']
            extrapolated_margin += actual_margin
            extrapolated_margin = extrapolated_margin.append(
                extrema).clip(-1, 1)
            extrapolated_margin.sort_values(inplace=True)

            dem_seats = (
                vote_share['democrat'] >= vote_share[
                    'republican']).value_counts()[True]

            seats_won = [seat for seat in range(
                self.num_districts[state] + 1)] + [
                self.num_districts[state]]
            percent_seats_won = [seat / self.num_districts[state]
                                 for seat in seats_won]

            for seats, margin in enumerate(
                    list(extrapolated_margin.values)):
                partisan_bias = (seats - 1) / self.num_districts[
                    state] - 0.5
                if margin >= 0:
                    break

            # use the step function to graph the Democratic share of
            # seats as total Democratic vote margin changes, per state
            axes.step(
                extrapolated_margin.values,
                percent_seats_won,
                alpha=stepalpha,
                color=self.colors[year],
                label=f"{year}; "
                      f"D{PartisanPlot.sign(round(partisan_bias * 100))}"
                      f" partisan bias",
                linewidth=3,
                where='post')
            axes.scatter(actual_margin,
                         dem_seats / self.num_districts[state],
                         s=dotsize,
                         color=self.colors[year])
            axes.legend()

    def __draw_default_parameters(self, default_parameters: List[Any]):
        """Assigns default parameters for draw(), because default
        arguments in Python are evaluated when the function is defined.

        Args:
            default_parameters: the values of the default parameters for
                draw(), in the list [axescolor, dotsize, axeslinestyle,
                axeslinewidth, stepalpha, num_yticks]
        """
        for i, var in enumerate(default_parameters):
            if var is None:
                default_parameters[i] = self.default_draw_values[i]
        return default_parameters

    @staticmethod
    def sign(num: int) -> str:
        """Adds positive sign in front of ``num``, if ``num`` is a
        positive number

        Args:
            num (int)

        Returns:
            (str): The string representation of ``num``, with a positive
            sign in front of ``num`` if it is positive
        """
        if num < 0:
            return str(num)
        return f"+{num}"
