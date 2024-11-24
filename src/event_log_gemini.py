import requests
import pandas as pd
import json
from datetime import datetime
import os
from typing import Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Predefined process descriptions
PROCESS_DESCRIPTIONS = {
    "1": {
        "description": "A hospital emergency room manages patient intake and treatment. The process starts when a patient arrives at the ER. First, the patient is registered. After registration, a triage nurse evaluates the patient to determine the severity of their condition. Depending on the triage assessment: If the condition is serious, the patient is immediately sent for treatment by a doctor. If the condition is non-serious, the patient waits until a doctor becomes available. After treatment begins: The doctor may request diagnostic tests (e.g., X-rays, bloodwork) in parallel with starting treatment. Once the test results are available, they are reviewed by the doctor to finalize treatment. If further follow-up is needed, the patient is scheduled for a follow-up appointment and discharged. Alternatively, if the treatment resolves the issue, the patient is directly discharged.",
        "suggested_params": {
            "num_traces": 25,
            "noise_level": 0.1,
            "uncommon_path_prob": 0.05,
            "missing_event_prob": 0.1
        }
    },
    "2": {
        "description": "A patient arrives at the hospital emergency department. They are first triaged by a nurse. Based on severity, they either wait in the waiting room or are taken directly to treatment. A doctor examines the patient and orders tests if necessary. After test results, treatment is provided. The patient is either discharged or admitted to the hospital.",
        "suggested_params": {
            "num_traces": 20,
            "noise_level": 0.15,
            "uncommon_path_prob": 0.1,
            "missing_event_prob": 0.05
        }
    }
}

def initialize_environment() -> str:
    """Initialize environment and return API key"""
    # Create output directory if it doesn't exist
    os.makedirs("../output", exist_ok=True)
    
    # First try to get API key from environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    
    # If not found in environment, prompt user
    if not api_key:
        api_key = input("Please enter your Gemini API key: ").strip()
        if not api_key:
            raise EnvironmentError("API key is required to proceed.")
    
    return api_key

def display_available_processes() -> None:
    """Displays available predefined process descriptions"""
    print("\nAvailable predefined processes:")
    print("-" * 50)
    for key, value in PROCESS_DESCRIPTIONS.items():
        print(f"\n{key}. Process Description:")
        print(f"{value['description'][:100]}...")
        print("\nSuggested Parameters:")
        for param, val in value['suggested_params'].items():
            print(f"- {param}: {val}")
    print("\n0. Enter custom process description")
    print("-" * 50)

def get_process_choice() -> tuple[str, Dict]:
    """Gets user choice for process description and returns description and parameters"""
    while True:
        choice = input("\nEnter your choice (0-2): ").strip()
        
        if choice == "0":
            print("\nEnter your process description (in paragraph form):")
            description = input().strip()
            if not description:
                print("Process description cannot be empty. Please try again.")
                continue
            params = get_custom_parameters()
            return description, params
            
        elif choice in PROCESS_DESCRIPTIONS:
            return (PROCESS_DESCRIPTIONS[choice]["description"], 
                   PROCESS_DESCRIPTIONS[choice]["suggested_params"])
        
        else:
            print("Invalid choice. Please try again.")

def get_custom_parameters() -> Dict:
    """Gets custom parameters from user input"""
    while True:
        try:
            num_traces = int(input("Enter the number of traces to simulate: "))
            params = {
                "num_traces": num_traces,
                "noise_level": float(input("Enter the noise level (0-1, e.g., 0.1 for 10%): ")),
                "uncommon_path_prob": float(input("Enter the uncommon path probability (0-1, e.g., 0.05 for 5%): ")),
                "missing_event_prob": float(input("Enter the missing event probability (0-1, e.g., 0.1 for 10%): "))
            }
            
            # Validate parameters
            if not (0 <= params["noise_level"] <= 1 and 
                   0 <= params["uncommon_path_prob"] <= 1 and 
                   0 <= params["missing_event_prob"] <= 1):
                raise ValueError
                
            return params
            
        except ValueError:
            print("Invalid input. Please enter valid numerical values.")
            print("Probabilities must be between 0 and 1.")

def fetch_gemini_response(process_description: str, num_traces: int, noise_level: float, 
                         uncommon_path_prob: float, missing_event_prob: float, api_key: str) -> str:
    """Fetches a response from the Gemini API using the given process description and simulation parameters."""
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [{
                "text": (
                    "Generate an event log for the following process description. "
                    "This event log will be used to construct a Petri net diagram, with the Alpha Algorithm applied to analyze it. "
                    "Ensure that the event log is Alpha Algorithm friendly, with clearly defined sequential, parallel, and looping events, "
                    "so that a Petri net can be easily generated. The event log must capture the following characteristics:\n\n"
                    "1. Include some noise: random, irrelevant events that do not belong to the actual process.\n"
                    "2. Simulate uncommon paths: rare but valid process variations.\n"
                    "3. ENSURE to NOT miss any CASE ID's.\n\n"
                    f"Process Description:\n{process_description}\n\n"
                    "Simulation Parameters:\n"
                    f"- Number of Traces: {num_traces}\n"
                    f"- Noise Level: {noise_level*100}%\n"
                    f"- Uncommon Path Probability: {uncommon_path_prob*100}%\n"
                    f"- Missing Event Probability: {missing_event_prob*100}%\n\n"
                    "Additional Instructions:\n"
                    "1. Alpha Algorithm will be applied on the event log so ensure to clearly identify the flow, loops, and parallel activities.\n"
                    "2. The event log should use timestamps in ISO 8601 format (e.g., '2023-11-20T14:23:00').\n"
                    f"3. Generate exactly {num_traces} traces with Case IDs sequentially numbered from 1 to {num_traces}.\n"
                    "4. The event log must have a tabular structure with three columns: 'Case ID,' 'Event,' and 'Timestamp.'\n\n"
                    "Return the event log in this exact format:\n"
                    "Case ID | Event | Timestamp\n\n"
                    "Ensure to generate all traces. DO NOT leave any Case IDs out."
                )
            }]
        }]
    }

    max_retries = 3
    for retry in range(max_retries):
        try:
            response = requests.post(gemini_url, headers={"Content-Type": "application/json"}, json=payload)
            response.raise_for_status()
            data = response.json()

            chatbot_response = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            ).strip()

            # Validate response
            if not chatbot_response:
                raise ValueError("Empty response received from API")

            return chatbot_response

        except (requests.RequestException, ValueError) as e:
            print(f"Attempt {retry + 1} failed: {e}")
            if retry < max_retries - 1:
                print("Retrying...")
                continue
            else:
                print("All retry attempts failed.")
                return None

def save_event_log_to_csv(chatbot_response: str, output_file: str, num_traces: int) -> None:
    """Saves the event log from the chatbot response to a CSV file."""
    try:
        rows = [line.strip() for line in chatbot_response.split("\n") if line.strip()]
        data = []
        
        for row in rows:
            if "|" in row:
                columns = [col.strip() for col in row.split("|")]
                if len(columns) == 3 and columns[0] != "Case ID":  # Skip header row
                    data.append(columns)
        
        df = pd.DataFrame(data, columns=["Case ID", "Event", "Timestamp"])
        
        # Convert Case ID to integers
        df['Case ID'] = pd.to_numeric(df['Case ID'], errors='coerce')
        
        # Check for missing Case IDs
        actual_case_ids = set(df['Case ID'].dropna().unique())
        expected_case_ids = set(range(1, num_traces + 1))
        missing_case_ids = expected_case_ids - actual_case_ids
        
        if missing_case_ids:
            print(f"Warning: Missing Case IDs detected: {missing_case_ids}")
            # Add placeholders for missing Case IDs
            for missing_id in missing_case_ids:
                df = pd.concat([df, pd.DataFrame({
                    "Case ID": [missing_id], 
                    "Event": ["Missing Trace"], 
                    "Timestamp": [datetime.now().isoformat()]
                })])
        
        # Sort by Case ID to ensure sequential order
        df = df.sort_values("Case ID").reset_index(drop=True)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Event log saved successfully to {output_file}.")
        
    except Exception as e:
        print(f"Error saving to CSV: {e}")

def main():
    try:
        # Initialize environment and get API key
        api_key = initialize_environment()
        
        # Display available processes
        display_available_processes()
        
        # Get process description and parameters
        process_description, params = get_process_choice()
        
        # Create output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"../output/event_log_gemini_{timestamp}.csv"
        
        # Fetch event log from Gemini
        print("\nFetching event log from Gemini API...")
        chatbot_response = fetch_gemini_response(
            process_description,
            params["num_traces"],
            params["noise_level"],
            params["uncommon_path_prob"],
            params["missing_event_prob"],
            api_key
        )
        
        if not chatbot_response:
            print("Failed to generate event log.")
            return
        
        # Save the fetched log to CSV
        save_event_log_to_csv(chatbot_response, output_file, params["num_traces"])
            
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()