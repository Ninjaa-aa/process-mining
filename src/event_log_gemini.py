import requests
import pandas as pd
import json
from datetime import datetime
import os
from typing import Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Predefined process descriptions with reduced trace counts
PROCESS_DESCRIPTIONS = {
    "1": {
        "description": "A customer submits an order online. The system validates the order and checks inventory. If items are available, payment processing begins. After successful payment, the warehouse receives a picking list and prepares the order. Quality check is performed before packaging. Finally, the order is shipped and customer receives tracking details.",
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
    """
    Initialize environment and return API key
    """
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Get API key from .env
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found in .env file. "
            "Please create a .env file with your API key: GEMINI_API_KEY=your_key_here"
        )
    
    return api_key

def display_available_processes() -> None:
    """
    Displays available predefined process descriptions
    """
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
    """
    Gets user choice for process description and returns description and parameters
    """
    while True:
        choice = input("\nEnter your choice (0-2): ").strip()
        
        if choice == "0":
            # Get custom process description
            print("\nEnter your process description (in paragraph form):")
            description = input().strip()
            if not description:
                print("Process description cannot be empty. Please try again.")
                continue
                
            # Get custom parameters
            params = get_custom_parameters()
            return description, params
            
        elif choice in PROCESS_DESCRIPTIONS:
            return (PROCESS_DESCRIPTIONS[choice]["description"], 
                   PROCESS_DESCRIPTIONS[choice]["suggested_params"])
        
        else:
            print("Invalid choice. Please try again.")

def get_custom_parameters() -> Dict:
    """
    Gets custom parameters from user input with restricted trace count
    """
    while True:
        try:
            num_traces = int(input("Enter the number of traces to simulate (20-30): "))
            if not (20 <= num_traces <= 30):
                print("Number of traces must be between 20 and 30.")
                continue

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
            print("Probabilities must be between 0 and 1, number of traces must be between 20 and 30.")


def fetch_gemini_response(process_description: str, num_traces: int, noise_level: float, 
                         uncommon_path_prob: float, missing_event_prob: float, api_key: str) -> str:
    """
    Fetches a response from the Gemini API using the given process description and simulation parameters.
    Modified for smaller batch sizes due to reduced trace count.
    """
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    
    # Break down the request into smaller batches
    batch_size = 5  # Process 5 traces at a time for better control
    num_batches = (num_traces + batch_size - 1) // batch_size
    all_responses = []
    
    for batch in range(num_batches):
        start_id = batch * batch_size + 1
        end_id = min((batch + 1) * batch_size, num_traces)
        current_batch_size = end_id - start_id + 1
        
        print(f"\nGenerating traces {start_id} to {end_id} (Batch {batch + 1}/{num_batches})...")
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": (
                        "Generate an event log for the following process description. "
                        "This event log will be used to construct a Petri net diagram, with the Alpha Algorithm applied to analyze it. "
                        "Ensure that the event log is Alpha Algorithm friendly, with clearly defined sequential, parallel, and looping events, "
                        f"so that a Petri net can be easily generated. Generate EXACTLY {current_batch_size} traces.\n\n"
                        f"Process Description:\n{process_description}\n\n"
                        "Simulation Parameters:\n"
                        f"- Generate traces for Case IDs from {start_id} to {end_id} ONLY\n"
                        f"- Noise Level: {noise_level*100}%\n"
                        f"- Uncommon Path Probability: {uncommon_path_prob*100}%\n"
                        f"- Missing Event Probability: {missing_event_prob*100}%\n\n"
                        "Requirements:\n"
                        f"1. MUST generate exactly {current_batch_size} traces\n"
                        f"2. MUST use Case IDs from {start_id} to {end_id} sequentially\n"
                        "3. Use timestamps in ISO 8601 format (e.g., '2023-11-20T14:23:00')\n"
                        "4. Include some noise events and uncommon paths based on the probabilities\n"
                        "5. Each trace must have at least 3 events\n\n"
                        "Return the event log in this exact format:\n"
                        "Case ID | Event | Timestamp\n\n"
                        f"Important: Generate EXACTLY {current_batch_size} traces with Case IDs from {start_id} to {end_id}."
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

                # Validate the response contains all expected Case IDs
                case_ids = set()
                for line in chatbot_response.split('\n'):
                    if '|' in line:
                        try:
                            case_id = int(line.split('|')[0].strip())
                            case_ids.add(case_id)
                        except (ValueError, IndexError):
                            continue

                case_ids.discard(0)  # Remove header row if present
                expected_ids = set(range(start_id, end_id + 1))

                if not expected_ids.issubset(case_ids):
                    if retry < max_retries - 1:
                        missing_ids = expected_ids - case_ids
                        print(f"Retry {retry + 1}: Missing Case IDs {missing_ids}, retrying...")
                        continue
                    else:
                        print(f"Warning: Could not generate complete traces for batch {batch + 1} after {max_retries} attempts")

                all_responses.append(chatbot_response)
                break

            except requests.RequestException as e:
                print(f"API request failed: {e}")
                if response is not None:
                    print(f"Response Status Code: {response.status_code}")
                    print(f"Response Content: {response.text}")
                if retry < max_retries - 1:
                    print(f"Retrying... (Attempt {retry + 2}/{max_retries})")
                    continue
                return None

    # Combine all responses
    combined_response = "Case ID | Event | Timestamp\n"  # Header
    for response in all_responses:
        # Skip header from subsequent responses
        lines = response.split('\n')
        if '|' in lines[0] and 'Case ID' in lines[0]:
            lines = lines[1:]
        combined_response += '\n'.join(lines) + '\n'

    return combined_response.strip()
def save_event_log_to_csv(chatbot_response: str, output_file: str, num_traces: int) -> None:
    """
    Saves the event log from the chatbot response to a CSV file.
    Ensures sequential Case IDs and handles missing traces.
    """
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
        output_path = os.path.join("output", output_file)
        df.to_csv(output_path, index=False)
        print(f"Event log saved successfully to {output_path}.")
    except Exception as e:
        print(f"Error saving to CSV: {e}")

def read_event_log_from_csv(input_file: str) -> pd.DataFrame:
    """
    Reads the event log from a CSV file into a pandas DataFrame.
    """
    try:
        input_path = os.path.join("output", input_file)
        df = pd.read_csv(input_path)
        print("Event log read successfully from CSV.")
        return df
    except Exception as e:
        print(f"Error reading from CSV: {e}")
        return None

if __name__ == "__main__":
    try:
        # Initialize environment and get API key
        api_key = initialize_environment()
        
        # Display available processes
        display_available_processes()
        
        # Get process description and parameters
        process_description, params = get_process_choice()
        
        # Output CSV file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = f"event_log_gemini_{timestamp}.csv"
        
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
        else:
            # Save the fetched log to a CSV file
            save_event_log_to_csv(chatbot_response, output_csv, params["num_traces"])
            
    except EnvironmentError as e:
        print(f"\nEnvironment Error: {e}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")