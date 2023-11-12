# Machine Learning for Battery Profit and Carbon Cutting Optimization

## Introduction
This project focuses on optimizing battery charging and discharging strategies to evaluate and optimize financial outcomes and reductions in carbon emissions. Utilizing a mix of data this model employs Long Short-Term Memory (LSTM) forecasting and rule-based algorithms to predict energy demands and optimize battery usage. It produces monthly forecasts and schedules for analysis.

## Features
- **Weather Forecasting**: Temperature, wind, and other significant weather patterns.
- **Emission Forecasting**: Predicting the distribution of various energy sources based on weather to create an emissions curve.
- **Energy Cost Forecasting**: Predicting the variable costs of electricity based on weather to create an energy cost curve.
- **Incentive Forecasting**: Predicting the payouts of incentive programs based on weather to create an incentive curve.
- **Carbon Accounting**: Assessing the impact of battery usage on the carbon emissions curve.
- **Profit Analysis**: Assessing the impact of battery usage on the energy cost and incentive curves.

## Data Sources
- Historical Weather Data from VisualCrossings.
- System Demand Data from ISO-NE.
- Energy Pricing Data from GridStatus.io.

## Technical Overview
The project leverages LSTM models for precise time-series forecasting and rule-based algorithms for optimization strategies, tailored to specific energy scenarios.

## Model Details
- **LSTM Model**: Used for forecasting customer load, system demand, daily dispatch schedule, and energy pricing.
- **Rule-Based Algorithms**: Employed for calculating profits and generating efficient discharge schedules.

## Installation and Usage
Detailed instructions on setting up the environment, installing dependencies, and running the model.

## Contact Information
Tyler Sims - tyler.jsims97@gmail.com
