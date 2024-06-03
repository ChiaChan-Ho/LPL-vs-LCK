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
We permformed univariate analysis on the total kills statistics in the dataset.

<iframe
  src="assets/total_kills_histogram.html"
  width="725"
  height="525"
  frameborder="0"
></iframe>

The histogram shows the distribution of total kills per game is roughly normal, with a slight right skew. This indicates that while most games have a typical number of kills, there are some with exceptionally high kill counts. This distribution is useful for our analysis, providing a basis for comparing the aggression levels between the LPL and LCK.

We also plot a graph for the dirstribution of damage per minute in the data set.



<iframe
  src="assets/damage_per_minute_histogram.html"
  width="725"
  height="525"
  frameborder="0"
></iframe>

The histogram shows the distribution of damage per minute (DPM) is roughly normal, with a slight right skew. This indicates that most games have a typical amount of damage dealt, but there are some instances with exceptionally high DPM. This distribution helps us understand the overall damage output in games, which will be useful for comparing the aggressiveness between the LPL and LCK.

### Bivariate Analysis
We performed bivariate analysis on the combined kills per minute (CKPM) statistics to compare the kill rates between the LPL and LCK leagues.

<iframe
  src="assets/average_ckpm_lpl_vs_lck.html"
  width="725"
  height="525"
  frameborder="0"
></iframe>

According to the plot, the LPL has a higher average CKPM compared to the LCK. Specifically, the LPL averages approximately 0.85 CKPM, while the LCK averages around 0.65 CKPM. This indicates that matches in the LPL tend to have more kills per minute than those in the LCK.

This analysis is important for our study as it provides a foundation for comparing the gameplay characteristics of the two leagues. Understanding these differences will be beneficial for our later analysis.

### Interesting Aggregates
Below are some interesting aggregates to investigate within the dataset. 

| ckpm > mean   |   deaths |   total kills |   minion kills |   penta kills |
|:--------------|---------:|--------------:|---------------:|--------------:|
| False         |   128523 |        128257 |    8.05211e+06 |            77 |
| True          |   169057 |        168683 |    6.32278e+06 |           134 |

We first grouped the cleaned dataset based on whether the combined kills per minute (CKPM) is above or below the mean CKPM. We then calculated the sum of various statistics for each group. By comparing the gaming statistics for matches with CKPM above the mean to those with CKPM below the mean, we gain better insights into how higher kill rates impact other game metrics.

The results show that matches with CKPM above the mean have better statistics: more total kills, penta kills, and deaths, but fewer minion kills. This analysis provides a foundation for understanding the relationship between kill rates and other gameplay metrics, which will be useful for our later analysis.

---

## Assessment of Missingness

### NMAR Analysis

In our data, we believe the column `url` is Not Missing At Random (NMAR). Upon examining the url column, we observe that there are no specific trends of missingness or any evidence suggesting that the missing URLs depend on other columns in the dataset. In the context of League of Legends matches, URLs are more likely to be recorded for high-profile or significant matches. This implies that the missingness of these URLs is inherently tied to the significance or profile of the match itself.

The missingness of the `url` column depends on the match's significance and availability, rather than on other observed data in our dataset. To make this column Missing at Random (MAR), we might obtain additional data such as `match importance`, which indicates the importance or profile level of the match, and `broadcast status`, which shows whether the match was broadcasted or uploaded online. These additional data points would help explain the missingness and potentially transform the missingness mechanism from NMAR to MAR.

### Missingness Dependency

In this part, we are going to test if the missingness of the `split` column depends on the league type. The two league types we used are major regions and wildcard regions. The significance level we chose for the permutation test is 0.05, and the test statistic is the difference in the proportion of missing values between major leagues and wildcard leagues.

**Null Hypothesis (H0)**: The proportion of missing values in the `split` column is the same for both major leagues and wildcard leagues. In other words, the distribution of missing values is independent of the league type.

**Alternative Hypothesis (H1)**: The proportion of missing values in the `split` column is different for major leagues compared to wildcard leagues. In other words, the distribution of missing values is dependent on the league type.

Below is the observed distribution of missing values in the `split` column between major and wildcard leagues, along with the permutation test outcomes.

<iframe
  src="assets/league_split_outcomes.html"
  width="725"
  height="525"
  frameborder="0"
></iframe>

The observed statistic for this permutation test is 0.2736, and the p-value is 0.0. Since the p-value is less than the 0.05 significance level, we reject the null hypothesis. Thus, the missingness of the `split` column depends on the league type.

The second part here, we are going to test if the missingness of the `split` column depends on the number of minion kills. The significance level we chose for the permutation test is 0.05, and the test statistic is the difference in the proportion of missing values between players with minion kills above the mean and those with minion kills below the mean.

**Null Hypothesis (H0)**: The proportion of missing values in the `split` column is the same for players with minion kills above the mean and for players with minion kills below the mean. In other words, the distribution of missing values is independent of the number of minion kills.

**Alternative Hypothesis (H1)**: The proportion of missing values in the `split` column is different for players with minion kills above the mean compared to players with minion kills below the mean. In other words, the distribution of missing values is dependent on the number of minion kills.

Below is the observed distribution of missing values in the `split` column between players with minion kills above the mean and those with minion kills below the mean, along with the permutation test outcomes.

<iframe
  src="assets/minion_split_outcomes.html"
  width="725"
  height="525"
  frameborder="0"
></iframe>

The observed statistic for this permutation test is 0.0114, and the p-value is 0.0981. Since the p-value is greater than the 0.05 significance level, we fail to reject the null hypothesis. Thus, the missingness of the `split` column does not depend on the number of minion kills.

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

