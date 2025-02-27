import pandas as pd
import requests

def main():
    # Load CSV into a DataFrame
    df = pd.read_csv("creditcard.csv")
    
    # Drop the "Class" column
    df = df.drop("Class", axis=1)
    
    # Choose a random row
    random_row = df.sample(n=1)
    
    # Convert the row to a list of features.
    # This will be a list of one list: [[feature1, feature2, ..., feature30]]
    features = random_row.values.tolist()
    
    # Construct the JSON payload
    payload = {"input": features}
    print("Payload:", payload)
    
    # Send a POST request to the local prediction server
    url = "http://localhost:8181/predict"
    response = requests.post(url, json=payload)
    
    print("Response:", response.json())

if __name__ == "__main__":
    main()
