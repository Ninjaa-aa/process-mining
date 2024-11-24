# Process Mining Analysis Project

## Team Members
- 22i2433 - Hammad Zahid
- 22i2514 - Maryam Khan
- 22i8764 - Emaan Shaikh
- 22i2617 - Umar Mahmood

## Overview
This project implements a comprehensive process mining system that combines event log generation, process model discovery using the Alpha Algorithm, and sophisticated model evaluation techniques. Our implementation provides tools for analyzing business processes, discovering process models, and evaluating their effectiveness.

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

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Graphviz (for visualization)
- Sufficient disk space for output files

### Installation

1. Clone the repository and create a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Unix/MacOS
python -m venv venv
source venv/bin/activate
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment:
```bash
# Create .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

## Core Features

### 1. Event Log Generation
```python
params = {
    "num_traces": 25,           # Number of traces to generate
    "noise_level": 0.1,         # Noise in the process (0-1)
    "uncommon_path_prob": 0.05, # Probability of uncommon paths
    "missing_event_prob": 0.1   # Probability of missing events
}

generator = EventLogGenerator(process_description, params)
event_log = generator.generate()
```

Key Features:
- Configurable trace generation
- Noise and variation control
- Built-in validation
- Standardized output format

### 2. Process Model Discovery
```python
alpha = AlphaAlgorithm(event_log)
petri_net = alpha.discover()
visualizer = PetriNetVisualizer()
visualizer.visualize(petri_net)
```

Capabilities:
- Alpha algorithm implementation
- Petri net generation
- Parallel activity detection
- Advanced pattern recognition

### 3. Model Evaluation
```python
evaluator = ProcessModelEvaluator(event_log, petri_net)
metrics = evaluator.evaluate()
report = evaluator.generate_report()
```

Metrics:
- Fitness assessment
- Precision analysis
- Generalization testing
- Detailed reporting

## Analysis Framework

### Data Preprocessing
- Event standardization
- Temporal ordering
- Noise filtering
- Case completion verification
- Format standardization

### Evaluation Metrics

1. Fitness Evaluation
   - Start/end validation
   - Transition conformance
   - Trace replay analysis
   - Score normalization

2. Precision Analysis
   - Behavioral pattern analysis
   - Control flow verification
   - Unused path detection
   - Structural validation

3. Generalization Assessment
   - Pattern extraction
   - Test case generation
   - Variant analysis
   - Cross-validation

## Dependencies

Core requirements:
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

## Usage Guidelines

1. Event Log Generation:
   - Configure process parameters
   - Review generated traces
   - Validate output completeness

2. Model Discovery:
   - Examine Petri net visualizations
   - Verify transition relationships
   - Review process patterns

3. Evaluation:
   - Analyze performance metrics
   - Review evaluation reports
   - Implement recommended improvements

## Support and Contact

For technical support or queries:
- Create an issue in the repository
- Consult the documentation
- Review error logs

## Contributing
We welcome contributions to improve the project. Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with a clear description of changes

---
For detailed implementation guides and examples, please refer to the documentation in the `docs/` directory.