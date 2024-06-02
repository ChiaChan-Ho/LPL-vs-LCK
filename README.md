# Playstyle Dynamics in Professional League of Legends: Aggressiveness in LPL vs. LCK

Comparative Analysis of Aggressiveness in LPL and LCK is a comprehensive data science project conducted at UCSD. This project involves various stages of analysis, from initial exploratory data analysis and hypothesis testing to the creation of baseline models and a thorough fairness analysis. Our main focus is to explore the differences in playstyles between the League of Legends Pro League (LPL) and the League of Legends Champions Korea (LCK), particularly in terms of aggressiveness.

by Chia-Chan Ho (chh061@ucsd.edu) and Yuwen Zhong (y4zhong@ucsd.edu)

---

## Introduction
### General Introduction
League of Legends (LoL) is a globally popular multiplayer online battle arena (MOBA) game created by Riot Games. With millions of players worldwide, it has become a significant force in the esports world. The dataset we are using for this project is sourced from Oracle's Elixir, which provides detailed match data from professional LoL esports games played throughout 2023.

This dataset offers a wealth of information, including crucial gameplay statistics and match outcomes, which can help us understand player behaviors, team strategies, and overall match dynamics. It encompasses various features such as individual player performance, team tactics, in-game metrics, and the general flow of matches.

Understanding the distinct playstyles of different leagues is beneficial for analysts, coaches, and fans alike. In particular, we are interested in examining the LPL and LCK, two of the most popular and competitive leagues in the world. There is a widespread belief among the esports community that the LPL tends to have a more aggressive, high-tempo playstyle, while the LCK is often seen as more methodical and strategic.

Our central question is: **Is the LPL demonstrably more aggressive than the LCK?** We aim to utilize data analysis techniques to compare the aggressiveness of teams from these two leagues and to develop a machine learning model capable of classifying a game's league based on gameplay statistics such as damage per minute (DPM) and combined kills per minute (CKPM). This analysis can provide significant insights that enhance strategic decision-making, team composition optimization, and overall appreciation of the game.

### Introduction of Columns
The dataset contains a comprehensive array of columns capturing gameplay metrics and match outcomes from professional League of Legends esports matches. The dataset consists of 125904 rows. Below are brief introduction to some key columns:

- `gameid`: A unique identifier for each match, allowing us to distinguish between different games.

- `league`: Specifies the league in which the match took place.

- `split`: Indicates the season split during which the match was played.

- `position`: The role or position played by an individual player within their team (e.g., top, jungle, mid, bot, support).

- `gamelength`: The total duration of the game in seconds.

- `teamkills`: The total number of kills secured by a team during the match.

- `pentakills`: The number of pentakills (a single player eliminating all five opponents) achieved by a player.

- `ckpm`: Combined kills per minute, indicating the average number of kills per minute in the game.

- `damagetochampions`: The total damage dealt to enemy champions.

- `dpm`: Damage per minute, indicating the average damage dealt to enemy champions per minute.

- `minionkills`: The number of minions killed by a player, reflecting their efficiency in farming.

- `deaths`: The total number of times a player or team was eliminated by the enemy.

- `url`: A link to the detailed match data or replay, providing additional context and information about each game.

---

## Data Cleaning and Exploratory Data Analysis
### Data Cleaning

To streamline our analysis, we first retain only the columns relevant to our study: `gameid`, `league`, `split`, `gamelength`, `teamkills`, `pentakills`, `ckpm`, `damagetochampions`, `dpm`, `minionkills`, `deaths`, and `url`. Each game in our dataset initially has 12 rows: 10 rows representing each player's statistics and 2 rows summarizing the overall team performance. For our analysis, we focus on the team summary rows, so we filter out the player-specific rows.

Furthermore, to ensure that each row in our dataframe represents a single match, we aggregate the relevant statistics. Specifically, we use the maximum values for columns such as `league`, `split`, `gamelength`, `ckpm`, and `url` to capture the overall match information. For the team-specific statistics, such as `teamkills`, `pentakills`, `damagetochampions`, `dpm`, `minionkills`, and `deaths`, we sum these values to get the total statistics for each game.

To enhance clarity, we also rename several columns: 
- `teamkills` to `total kills`

- `minionkills` to `minion kills`

- `damagetochampions` to `damage to champions`

- `pentakills` to `penta kills`

- `gamelength` to `game length`

The resulting cleaned dataset contains one row per match, with columns representing aggregated statistics that will be used in our analysis.

Below is the head of our resulting dataframe.

| gameid             | league   |   split |   game length |   total kills |   penta kills |   ckpm |   damage to champions |     dpm |   minion kills |   deaths | url                                          |
|:-------------------|:---------|--------:|--------------:|--------------:|--------------:|-------:|----------------------:|--------:|---------------:|---------:|:---------------------------------------------|
| 10000-10000_game_1 | LDL      |     nan |          1704 |            25 |             0 | 0.8803 |                109350 | 3850.35 |              0 |       25 | https://lpl.qq.com/es/stats.shtml?bmid=10000 |
| 10000-10000_game_2 | LDL      |     nan |          1809 |            36 |             0 | 1.194  |                172658 | 5726.63 |              0 |       36 | https://lpl.qq.com/es/stats.shtml?bmid=10000 |
| 10000-10000_game_3 | LDL      |     nan |          1738 |            39 |             0 | 1.3464 |                148264 | 5118.43 |              0 |       39 | https://lpl.qq.com/es/stats.shtml?bmid=10000 |
| 10000-10000_game_4 | LDL      |     nan |          1844 |            24 |             0 | 0.7809 |                138964 | 4521.61 |              0 |       24 | https://lpl.qq.com/es/stats.shtml?bmid=10000 |
| 10000-10000_game_5 | LDL      |     nan |          1691 |            28 |             0 | 0.9935 |                110761 | 3930.02 |              0 |       28 | https://lpl.qq.com/es/stats.shtml?bmid=10000 |

*Note*: The cleaned dataset now consists of all the columns we need for both hypothesis testing and prediction modeling. Further adjustments will be made as necessary in subsequent sections.

### Univariate Analysis
We permformed univariate analysis on the total kills statistics in the dataset

<iframe
  src="assets/total_kills_histogram.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

This histogram shows that...

We also plot a graph for the dirstribution of damage per minute in the data set.

<iframe
  src="assets/damage_per_minute_histogram.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

This histogram shows that...

---

## Assessment of Missingness

Here's what a Markdown table looks like. Note that the code for this table was generated _automatically_ from a DataFrame, using

```py
print(cleaned_df[['league', 'split']].head().to_markdown(index=False))
```

| league   | split   |
|:---------|:--------|
| LFL2     | Spring  |
| LFL2     | Spring  |
| LFL2     | Spring  |
| LFL2     | Spring  |
| LFL2     | Spring  |

---

## Hypothesis Testing

---

## Framing  a Prediction Problem

---

## Baseline Model

---

## Final Model

---

## Fairness Analysis

---

