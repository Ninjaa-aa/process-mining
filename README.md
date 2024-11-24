# Process Mining Analysis Project

This project implements comprehensive process mining techniques including event log generation, process model discovery using the Alpha Algorithm, and model evaluation.

## Project Structure
```
process-mining/
├── src/
│   ├── generator.py      # Event log generation with controllable parameters
│   ├── alpha_algo.py     # Alpha algorithm implementation
│   └── evaluation.py     # Model evaluation metrics
├── output/               # Generated files
├── docs/                # Documentation
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Setup and Installation

1. Create and activate virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Unix/MacOS
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create .env file with API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Running the Project

1. Generate Event Log:
```bash
python src/generator.py
```
Features:
- Process description input (predefined or custom)
- Configurable parameters:
  * Number of traces (20-30 recommended)
  * Noise level
  * Uncommon path probability
  * Missing event probability
- Outputs standardized CSV format

2. Discover Process Model:
```bash
python src/alpha_algo.py
```
Features:
- Applies Alpha algorithm
- Generates Petri net visualization
- Handles parallel activities and choices

3. Evaluate Model:
```bash
python src/evaluation.py
```
Features:
- Calculates fitness, precision, and generalization
- Generates detailed evaluation report
- Provides improvement recommendations

## Process Analysis Report

### 1. Preprocessing Steps

#### Event Log Generation
- Structured input processing
- Controlled trace generation
- Timestamp management
- Built-in validation
- Pattern enforcement
- Error handling

#### Data Cleaning
- Event name standardization
- Sequential ordering
- Noise filtering
- Case completion verification
- Format standardization

### 2. Design Choices

#### Alpha Algorithm Implementation
- Footprint matrix construction
- Relationship pattern discovery
- Parallel activity detection
- Loop handling
- Transition validation

#### Evaluation Framework
- Comprehensive metrics
- Detailed analysis
- Pattern recognition
- Conformance checking
- Performance optimization

### 3. Model Evaluation

#### Metrics Implementation

1. Fitness Evaluation
   - Start/end validation
   - Transition conformance
   - Trace replay
   - Score normalization
   - Issue identification

2. Precision Analysis
   - Behavioral analysis
   - Control flow verification
   - Unused path detection
   - Structural validation
   - Pattern matching

3. Generalization Testing
   - Pattern extraction
   - Test trace generation
   - Variant analysis
   - Cross-validation
   - Flexibility assessment

### 4. Technical Implementation

#### Event Log Generator
```python
# Configure parameters
params = {
    "num_traces": 25,           # Number of traces to generate
    "noise_level": 0.1,         # Noise in the process (0-1)
    "uncommon_path_prob": 0.05, # Probability of uncommon paths
    "missing_event_prob": 0.1   # Probability of missing events
}

# Generate event log
generator = EventLogGenerator(process_description, params)
event_log = generator.generate()
```

#### Alpha Algorithm
```python
# Discover process model
alpha = AlphaAlgorithm(event_log)
petri_net = alpha.discover()
visualizer = PetriNetVisualizer()
visualizer.visualize(petri_net)
```

#### Model Evaluation
```python
# Evaluate model
evaluator = ProcessModelEvaluator(event_log, petri_net)
metrics = evaluator.evaluate()
report = evaluator.generate_report()
```

## Dependencies

Core dependencies:
```text
pandas==2.1.4
numpy==1.24.3
requests==2.31.0
python-dateutil==2.8.2
python-dotenv==1.0.0
graphviz==0.20.1
tabulate==0.9.0
tqdm==4.66.1
colorama==0.4.6
```

## Installation Notes

1. System Requirements:
   - Python 3.8+
   - Graphviz (for visualization)
   - Sufficient disk space for outputs

2. Environment Setup:
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate (Windows)
   venv\Scripts\activate
   
   # Activate (Unix/MacOS)
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. Configuration:
   - Set up .env file
   - Configure output directories
   - Adjust parameters as needed

## Usage Notes

1. Event Log Generation:
   - Review process description
   - Adjust parameters based on needs
   - Verify output completeness

2. Model Discovery:
   - Check Petri net visualization
   - Verify transitions
   - Review relationships

3. Evaluation:
   - Analyze metrics
   - Review detailed report
   - Consider recommendations

## Contact & Support

For issues and suggestions:
- Open an issue in the repository
- Review existing documentation
- Check error logs"# process-mining" 
