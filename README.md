# Modeling Project

A machine learning project for rain prediction using meteorological data from Israeli weather stations.

## Project Overview

This project develops binary classification models to predict rainfall events using historical weather data from Beit Dagan and Jerusalem stations (2010-2025).

## Project Structure

```
Modeling_Project/
├── Data/                          # Weather data (gitignored, available as zip in releases)
│   ├── Beit Dagan/
│   │   └── BD_P_RH_T_windDir_windSpeed_windDirSTD_rain_20101001_20250930.csv
│   └── Jerusalem/
│       └── jer_rain_20101001_20250930.csv
├── evaluation/
│   ├── evaluation_binary.py       # Model evaluation functions
│   └── __init__.py
├── model/
│   ├── binary_model.py            # Binary classification model
│   └── __init__.py
├── preprocessing/
│   ├── preprocessing.py           # Data preprocessing and feature engineering
│   └── __init__.py
├── main.ipynb                     # Main execution notebook (in Colab)
├── .gitignore
└── README.md
```

## Data

The weather data consists of:
- **Beit Dagan Station**: Pressure, Relative Humidity, Temperature, Wind Direction, Wind Speed, Wind Direction STD, and Rainfall
- **Jerusalem Station**: Rainfall data
- **Time Period**: October 1, 2010 - September 30, 2025
- **File Size**: ~40 MB per CSV

**Note**: CSV files are not tracked in git. Download the data zip file from the GitHub releases section.

## How to Run

This project is designed to run in **Google Colab**:

1. **Mount Google Drive**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Navigate to project**:
   ```python
   %cd /content/drive/MyDrive/Modeling_Project
   ```

3. **Open and run** `main.ipynb` in Google Colab

4. The notebook imports functions from the `.py` modules in `preprocessing/`, `model/`, and `evaluation/`

## Development Workflow

We develop in Jupyter notebooks (`.ipynb`) in Google Colab and maintain corresponding Python modules (`.py`) for version control and imports.

**Typical workflow**:
1. Develop/test code in notebooks (e.g., `preprocessing.ipynb`)
2. Manually copy working code to corresponding `.py` file (e.g., `preprocessing/preprocessing.py`)
3. Test imports in `main.ipynb`
4. Commit and push `.py` files to GitHub

## Git Workflow

### Daily Updates

When you're ready to save your work to GitHub:

```bash
# In Colab, mount Drive and navigate to project
cd /content/drive/MyDrive/Modeling_Project

# Get latest changes from collaborator
git pull

# Check what files you changed
git status

# Add your changed files
git add <filename>              # Add specific file
# OR
git add .                       # Add all changed files

# Commit with a descriptive message
git commit -m "Brief description of what you changed"

# Push to GitHub
git push
```

### Example Commit Messages

Good commit messages are clear and concise:
- ✅ `"Added feature scaling to preprocessing"`
- ✅ `"Fixed bug in model evaluation metrics"`
- ✅ `"Updated binary model with XGBoost"`
- ❌ `"Updated files"` (too vague)
- ❌ `"stuff"` (not descriptive)

### Before Starting Work

Always pull the latest changes first:
```bash
cd /content/drive/MyDrive/Modeling_Project
git pull
```

This ensures you have your collaborator's latest changes.

## Dependencies

Key Python libraries used:
- pandas
- numpy
- scikit-learn
- matplotlib
- (Add others as you use them)

Install in Colab:
```python
!pip install <package-name>
```

## Authors

- [Your Name]
- [Collaborator Name]

## License

[Add license if applicable]
