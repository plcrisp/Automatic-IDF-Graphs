
# ☔️ Precipitation Analysis: Automatic IDF Graphs

A Python tool for generating and visualizing Intensity-Duration-Frequency (IDF) curves based on rainfall data. This project helps hydrologists, engineers, and researchers analyze precipitation patterns, design drainage systems, and make informed decisions for flood risk management.

Whether you’re conducting research or planning stormwater systems, this project simplifies complex IDF calculations and provides valuable insights through clear, interactive visualizations.




## 🚀 Features

- **Automatic IDF Curve Generation**: Computes intensity-duration-frequency relationships for different return periods.

- **Data Cleaning & Processing**: Fixes errors, fills missing values, and removes outliers.

- **Flexible Time Interval Aggregation**: Handles sub-daily and daily precipitation data.

- **Extreme Precipitation Analysis**: Identifies annual maxima and sub-daily extreme rainfall events.

- **Interactive & High-Quality Visualizations**: Generates clear graphs for precipitation patterns and IDF curves.

- **Customizable Parameters**: Supports adjustments for different statistical distributions.

- **Multi-Source Data Support**: Compatible with datasets from INMET, CEMADEN, and other meteorological agencies.

- **Modular & Scalable Code**: Organized structure for easy customization and extension.


## 🏗️ Project Structure

This project is organized into well-defined folders to streamline data processing, analysis, and visualization of rainfall events. Below is an overview of each folder and its purpose.

```bash
📂 Precipitation-Analysis-IDF
│── datasets/            # Raw or pre-processed meteorological data
│── graphs/              # Generated IDF curves and precipitation event visualizations
│── parameters/          # Configuration files for model parameters
│── results/             # Final analysis results, processed reports, and statistics
│── utils/               # Utility functions for rainfall data handling
│── scripts/             # Core Python scripts for data processing and visualization
│── README.md            # Project documentation
```

### 📂 datasets

Stores raw or pre-processed data, such as rainfall measurements from sources such as INMET and CEMADEN.

### 📂 graphs

Contains the generated graphs, including IDF curves and visualizations of extreme precipitation events.

### 📂 parameters

Configuration files with model parameters, such as adjustments for distributions.

### 📂 results

Final analysis results such as sub-daily maximum tables, precipitation statistics and processed reports.

### 📂 utils

Module with auxiliary functions organized for handling and analyzing rainfall data.


## 🛠️ Core Scripts

- 🔄 *get_datasets.py:* Loads and prepares meteorological station data (e.g., INMET, CEMADEN), ensuring consistent structure for analysis.

- ⚙️ *data_processing.py*: Handles raw data manipulation: aggregation by intervals, saving to CSV, file reading, and generating precipitation distribution plots.

- 🔧  *error_correction.py*: Ensures data quality by fixing dates, filling missing values, and removing outliers for clean, reliable data.

- 🔬 *quality_analysis.py*: Evaluates data quality through correlation, consistency, and trend tests to validate precipitation time series.

- ⏱️ *intervals_manipulation.py*: Manages aggregation and disaggregation of precipitation data for different time intervals, essential for sub-daily intensity calculations.

- 🌧️ *extreme_precipitation_analysis.py*: Analyzes extreme precipitation events, calculating annual maxima and sub-daily extremes for easier analysis of intense rainfall.

- 📈 *extreme_precipitation_visualization.py*: Generates visualizations for interpreting extreme precipitation events, with optimized graphs to reveal rainfall patterns.

- 📊 *get_distribution.py*: analyze daily or sub-daily precipitation data by fitting statistical distributions, generating visualizations, and saving the parameters of the best-fitted distributions for further use.

- 📐 *idf_generator.py* : Generates IDF (Intensity-Duration-Frequency) curves for hydrological analysis. Computes key parameters (t0, n, K, m) by fitting statistical distributions, optimizing constants, and modeling relationships between rainfall intensity, duration, and frequency. Supports tasks like calculating theoretical precipitation for return periods and generating IDF tables, with optional CSV export for further analysis.



## 📖 References

- Instituto Nacional de Meteorologia (INMET) - https://www.inmet.gov.br

- Centro Nacional de Monitoramento e Alertas de Desastres Naturais (CEMADEN) - http://www.cemaden.gov.br
## 🤝 Contributing

Contributions are welcome! Feel free to submit issues, feature requests, or pull requests to improve the project.

🛠 **How to Contribute**

    1. Fork the repository

    2. Create a new branch (git checkout -b feature-new-feature)

    3. Commit your changes (git commit -m "Add a new feature")

    4. Push to the branch (git push origin feature-new-feature)

    5. Open a pull request
## 📬 Contact


For questions or suggestions, reach out via email or open an issue in the repository.

- 📧 Email: pedrolcrisp@gmail.com
- 🐙 GitHub: plcrisp
