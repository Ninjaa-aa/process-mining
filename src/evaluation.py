
import pandas as pd
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set
from datetime import datetime
import random

class ProcessModelEvaluator:
    def __init__(self, event_log_file: str):
        self.event_log_file = event_log_file
        self.event_log = None
        self.traces = []
        self.unique_events = set()
        self.model_stats = {
            'total_traces': 0,
            'unique_traces': 0,
            'total_events': 0,
            'unique_events': 0,
            'start_activities': 0,
            'end_activities': 0,
            'transitions': 0
        }
        self.metrics = {
            'fitness': 0.0,
            'precision': 0.0,
            'generalization': 0.0
        }
        self.issues = {
            'fitness': [],
            'precision': [],
            'generalization': []
        }
        self.valid_start_events = {'PatientArrival'}
        self.valid_end_events = {'Discharge', 'Admit'}
        
        # Define valid transitions for hospital process
        self.valid_transitions = {
            'PatientArrival': {'NurseTriage'},
            'NurseTriage': {'WaitRoom', 'DirectToTreatment'},
            'WaitRoom': {'DoctorExam'},
            'DirectToTreatment': {'DoctorExam'},
            'DoctorExam': {'OrderTests', 'Treatment'},
            'OrderTests': {'TestResults'},
            'TestResults': {'Treatment'},
            'Treatment': {'Discharge', 'Admit'},
            'Discharge': set(),
            'Admit': set()
        }
        
        self.load_and_preprocess()
        self.calculate_model_stats()

    def standardize_event_name(self, event: str) -> str:
        """Standardize event names."""
        name_mappings = {
            'NurseTriaging': 'NurseTriage',
            'WaitInWaitingRoom': 'WaitRoom',
            'WaitingRoom': 'WaitRoom',
            'DoctorExamination': 'DoctorExam',
            'TreatmentProvided': 'Treatment',
            'AdmitToHospital': 'Admit',
            'Admission': 'Admit'
        }
        return name_mappings.get(event, event)

    def load_and_preprocess(self):
        """Load and preprocess event log."""
        try:
            # Load CSV file
            self.event_log = pd.read_csv(self.event_log_file)
            
            # Remove placeholder events
            self.event_log = self.event_log[self.event_log['Event'] != '--------']
            
            # Standardize event names
            self.event_log['Event'] = self.event_log['Event'].apply(self.standardize_event_name)
            
            # Sort by Case ID and Timestamp
            self.event_log = self.event_log.sort_values(['Case ID', 'Timestamp'])
            
            # Extract traces
            traces_dict = {}
            for _, row in self.event_log.iterrows():
                case_id = row['Case ID']
                event = row['Event']
                if case_id not in traces_dict:
                    traces_dict[case_id] = []
                traces_dict[case_id].append(event)
            
            self.traces = list(traces_dict.values())
            self.unique_events = set(self.event_log['Event'].unique())
            
            print(f"Successfully loaded event log.")
            
        except Exception as e:
            print(f"Error loading event log: {e}")

    def calculate_model_stats(self):
        """Calculate model statistics."""
        self.model_stats['total_traces'] = len(self.traces)
        self.model_stats['unique_traces'] = len(set(tuple(trace) for trace in self.traces))
        self.model_stats['total_events'] = len(self.event_log)
        self.model_stats['unique_events'] = len(self.unique_events)
        
        # Calculate start and end activities
        start_activities = set(trace[0] for trace in self.traces)
        end_activities = set(trace[-1] for trace in self.traces)
        self.model_stats['start_activities'] = len(start_activities)
        self.model_stats['end_activities'] = len(end_activities)
        
        # Calculate transitions
        transitions = set()
        for trace in self.traces:
            for i in range(len(trace) - 1):
                transitions.add((trace[i], trace[i + 1]))
        self.model_stats['transitions'] = len(transitions)

    def evaluate_fitness(self) -> float:
        """Calculate fitness of the process model."""
        total_score = 0
        total_traces = len(self.traces)
        
        for trace in self.traces:
            score = 0
            issues = []
            
            # Check start event
            if trace[0] not in self.valid_start_events:
                issues.append(f"Invalid start event: {trace[0]}")
            else:
                score += 1
            
            # Check end event
            if trace[-1] not in self.valid_end_events:
                issues.append(f"Invalid end event: {trace[-1]}")
            else:
                score += 1
            
            # Check transitions
            valid_transitions = 0
            for i in range(len(trace) - 1):
                if trace[i+1] in self.valid_transitions.get(trace[i], set()):
                    valid_transitions += 1
                else:
                    issues.append(f"Invalid transition: {trace[i]} -> {trace[i+1]}")
            
            if len(trace) > 1:
                transition_score = valid_transitions / (len(trace) - 1)
                score += transition_score
            
            trace_fitness = score / 3  # Normalize by three checks (start, end, transitions)
            total_score += trace_fitness
            
            if issues:
                self.issues['fitness'].append({
                    'trace': ' -> '.join(trace),
                    'score': trace_fitness,
                    'issues': issues
                })
        
        fitness = total_score / total_traces if total_traces > 0 else 0
        self.metrics['fitness'] = fitness
        return fitness

    def evaluate_precision(self) -> float:
        """Calculate precision of the process model."""
        # Get observed transitions from log
        observed_transitions = set()
        for trace in self.traces:
            for i in range(len(trace) - 1):
                observed_transitions.add((trace[i], trace[i + 1]))
        
        # Get all possible transitions from model
        possible_transitions = set()
        for event in self.unique_events:
            for next_event in self.valid_transitions.get(event, set()):
                possible_transitions.add((event, next_event))
        
        # Calculate precision
        if not possible_transitions:
            return 0.0
            
        precision = len(observed_transitions) / len(possible_transitions)
        self.metrics['precision'] = precision
        
        # Record unused transitions
        unused = possible_transitions - observed_transitions
        for transition in unused:
            self.issues['precision'].append({
                'transition': f"{transition[0]} -> {transition[1]}",
                'type': 'Unused transition'
            })
        
        return precision

    def evaluate_generalization(self, num_test_traces: int = 20) -> float:
        """Calculate generalization of the process model."""
        # Generate test traces
        test_traces = []
        attempts = 0
        max_attempts = 100
        
        while len(test_traces) < num_test_traces and attempts < max_attempts:
            trace = self.generate_trace()
            if trace:
                test_traces.append(trace)
            attempts += 1
        
        # Extract patterns
        original_patterns = self.extract_patterns(self.traces)
        test_patterns = self.extract_patterns(test_traces)
        
        # Calculate generalization
        shared_patterns = original_patterns & test_patterns
        total_patterns = original_patterns | test_patterns
        
        generalization = len(shared_patterns) / len(total_patterns) if total_patterns else 0
        self.metrics['generalization'] = generalization
        return generalization

    def generate_trace(self) -> List[str]:
        """Generate a single valid trace."""
        trace = [random.choice(list(self.valid_start_events))]
        max_length = 10
        attempts = 0
        max_attempts = 20
        
        while len(trace) < max_length and attempts < max_attempts:
            current = trace[-1]
            next_possible = list(self.valid_transitions.get(current, set()))
            
            if not next_possible:
                if current in self.valid_end_events:
                    return trace
                break
            
            next_event = random.choice(next_possible)
            trace.append(next_event)
            attempts += 1
            
            if next_event in self.valid_end_events and len(trace) >= 3:
                return trace
        
        return None

    def extract_patterns(self, traces: List[List[str]]) -> Set[Tuple[str, ...]]:
        """Extract patterns from traces."""
        patterns = set()
        for trace in traces:
            # Extract pairs
            for i in range(len(trace) - 1):
                patterns.add((trace[i], trace[i + 1]))
            # Extract triplets
            for i in range(len(trace) - 2):
                patterns.add((trace[i], trace[i + 1], trace[i + 2]))
        return patterns

    def generate_evaluation_report(self) -> str:
        """Generate comprehensive evaluation report."""
        report = []
        report.append("\nProcess Model Evaluation Report")
        report.append("=" * 50)
        
        # Model Statistics
        report.append("\n1. Model Statistics")
        report.append("-" * 20)
        report.append(f"Total traces: {self.model_stats['total_traces']}")
        report.append(f"Unique traces: {self.model_stats['unique_traces']}")
        report.append(f"Total events: {self.model_stats['total_events']}")
        report.append(f"Unique events: {self.model_stats['unique_events']}")
        report.append(f"Start activities: {self.model_stats['start_activities']}")
        report.append(f"End activities: {self.model_stats['end_activities']}")
        report.append(f"Transitions: {self.model_stats['transitions']}")
        
        # Metrics
        report.append("\n2. Conformance Metrics")
        report.append("-" * 20)
        report.append(f"Fitness: {self.metrics['fitness']:.3f}")
        report.append(f"Precision: {self.metrics['precision']:.3f}")
        report.append(f"Generalization: {self.metrics['generalization']:.3f}")
        
        # Analysis
        report.append("\n3. Detailed Analysis")
        report.append("-" * 20)
        
        # Fitness Issues
        if self.issues['fitness']:
            report.append("\nFitness Issues:")
            for issue in self.issues['fitness'][:5]:
                report.append(f"- Trace: {issue['trace']}")
                report.append(f"  Score: {issue['score']:.3f}")
                report.append(f"  Issues: {', '.join(issue['issues'])}")
        
        # Precision Issues
        if self.issues['precision']:
            report.append("\nPrecision Issues:")
            report.append(f"Found {len(self.issues['precision'])} unused transitions:")
            for issue in self.issues['precision'][:5]:
                report.append(f"- {issue['transition']}")
        
        # Recommendations
        report.append("\n4. Recommendations")
        report.append("-" * 20)
        report.extend(self.generate_recommendations())
        
        return "\n".join(report)

    def generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        if self.metrics['fitness'] < 0.8:
            recommendations.append("- Improve fitness by:")
            recommendations.append("  * Standardizing event names")
            recommendations.append("  * Ensuring valid start/end events")
            recommendations.append("  * Adding missing transition paths")
        
        if self.metrics['precision'] < 0.7:
            recommendations.append("- Improve precision by:")
            recommendations.append("  * Removing unused pathways")
            recommendations.append("  * Adding constraints to control flow")
            recommendations.append("  * Reviewing parallel activities")
        
        if self.metrics['generalization'] < 0.6:
            recommendations.append("- Improve generalization by:")
            recommendations.append("  * Adding flexibility for variants")
            recommendations.append("  * Reviewing strict sequences")
            recommendations.append("  * Considering more parallel paths")
        
        return recommendations

def main():
    # Initialize evaluator
    print("\nInitializing Process Model Evaluator...")
    evaluator = ProcessModelEvaluator("output/event_log_gemini.csv")
    
    # Print model statistics
    print("\nCalculating model statistics...")
    print(f"Total traces: {evaluator.model_stats['total_traces']}")
    print(f"Unique events: {evaluator.model_stats['unique_events']}")
    
    # Calculate metrics
    print("\nCalculating conformance metrics...")
    fitness = evaluator.evaluate_fitness()
    precision = evaluator.evaluate_precision()
    generalization = evaluator.evaluate_generalization()
    
    print(f"\nResults:")
    print(f"Fitness: {fitness:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Generalization: {generalization:.3f}")
    
    # Generate and save report
    print("\nGenerating evaluation report...")
    report = evaluator.generate_evaluation_report()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"output/evaluation_report_{timestamp}.txt"
    
    with open(report_file, "w", encoding='utf-8') as f:
        f.write(report)
        
    print(f"\nEvaluation report saved to: {report_file}")

if __name__ == "__main__":
    main()
