import pandas as pd
import json


file_path = "Bicycle_Thefts_data.csv"
data = pd.read_csv(file_path)

columns = [
    "BIKE_MAKE",
    "BIKE_MODEL",
    "BIKE_TYPE",
    "BIKE_SPEED",
    "BIKE_COLOUR",
    "PREMISES_TYPE",
    "LOCATION_TYPE",
    "HOOD_140",
    "NEIGHBOURHOOD_140"
]

unique_values = {column: data[column].dropna().unique().tolist() for column in columns}

data = {}
for column, values in unique_values.items():
    data[column] = values

output_file = "../api/unique_values.json"
with open(output_file, "w") as f:
    json.dump(data, f, indent=4)