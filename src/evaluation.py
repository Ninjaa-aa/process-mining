import pandas as pd
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set, Optional
from datetime import datetime
import networkx as nx
import os
import re

class ProcessModelEvaluator:
    def __init__(self, event_log_file: str):
        """Initialize the Process Model Evaluator with an event log file."""
        self.event_log_file = event_log_file
        self.event_log = None
        self.traces = []
        self.unique_events = set()
        self.model_stats = defaultdict(int)
        self.metrics = defaultdict(float)
        self.issues = defaultdict(list)
        self.process_graph = nx.DiGraph()
        
        # Load and process the event log
        self.load_and_preprocess()
        self.build_process_graph()
        self.calculate_model_stats()

    def clean_timestamp(self, timestamp_str: str) -> str:
        """Clean timestamp string by removing noise indicators and extra whitespace."""
        # Remove "(Noise Event)" and any extra whitespace
        cleaned = re.sub(r'\s*\(Noise Event\)\s*', '', str(timestamp_str)).strip()
        return cleaned

    def load_and_preprocess(self) -> None:
        """Load and preprocess the event log from CSV."""
        try:
            # Load CSV file
            self.event_log = pd.read_csv(self.event_log_file)
            
            # Basic data cleaning
            self.event_log = self.event_log.dropna()
            self.event_log = self.event_log[self.event_log['Event'].str.strip() != '']
            
            # Clean timestamps
            self.event_log['Timestamp'] = self.event_log['Timestamp'].apply(self.clean_timestamp)
            
            # Convert timestamps with error handling
            try:
                self.event_log['Timestamp'] = pd.to_datetime(
                    self.event_log['Timestamp'],
                    format='mixed',  # Allow mixed formats
                    errors='coerce'  # Replace invalid timestamps with NaT
                )
            except Exception as e:
                print(f"Warning: Error converting timestamps: {e}")
                print("Attempting alternative timestamp parsing...")
                try:
                    # Try parsing with a more flexible approach
                    self.event_log['Timestamp'] = pd.to_datetime(
                        self.event_log['Timestamp'],
                        infer_datetime_format=True,
                        errors='coerce'
                    )
                except Exception as e:
                    print(f"Error: Failed to parse timestamps: {e}")
                    raise
            
            # Remove rows with invalid timestamps
            invalid_timestamps = self.event_log['Timestamp'].isna()
            if invalid_timestamps.any():
                print(f"Warning: Removed {invalid_timestamps.sum()} rows with invalid timestamps")
                self.event_log = self.event_log.dropna(subset=['Timestamp'])
            
            # Sort by Case ID and Timestamp
            self.event_log = self.event_log.sort_values(['Case ID', 'Timestamp'])
            
            # Extract traces and unique events
            traces_dict = defaultdict(list)
            for _, row in self.event_log.iterrows():
                traces_dict[row['Case ID']].append(row['Event'])
            
            self.traces = list(traces_dict.values())
            self.unique_events = set(self.event_log['Event'].unique())
            
            print(f"Successfully loaded event log with {len(self.traces)} traces.")
            print(f"Found {len(self.unique_events)} unique events.")
            
        except Exception as e:
            print(f"Error loading event log: {e}")
            raise
    
    def build_process_graph(self) -> None:
        """Build a directed graph representation of the process."""
        # Add nodes for all events
        for event in self.unique_events:
            self.process_graph.add_node(event, frequency=0)
        
        # Add edges and calculate frequencies
        edge_counts = defaultdict(int)
        node_counts = defaultdict(int)
        
        for trace in self.traces:
            for i, event in enumerate(trace):
                node_counts[event] += 1
                if i < len(trace) - 1:
                    edge_counts[(event, trace[i + 1])] += 1
        
        # Update node frequencies
        for event, count in node_counts.items():
            self.process_graph.nodes[event]['frequency'] = count
        
        # Add weighted edges
        for (source, target), count in edge_counts.items():
            self.process_graph.add_edge(source, target, weight=count)

    def calculate_model_stats(self) -> None:
        """Calculate comprehensive model statistics."""
        self.model_stats.update({
            'total_traces': len(self.traces),
            'unique_traces': len(set(tuple(trace) for trace in self.traces)),
            'total_events': len(self.event_log),
            'unique_events': len(self.unique_events),
            'avg_trace_length': sum(len(trace) for trace in self.traces) / len(self.traces),
            'min_trace_length': min(len(trace) for trace in self.traces),
            'max_trace_length': max(len(trace) for trace in self.traces)
        })
        
        # Calculate start and end activities
        start_activities = set(trace[0] for trace in self.traces)
        end_activities = set(trace[-1] for trace in self.traces)
        self.model_stats.update({
            'start_activities': len(start_activities),
            'end_activities': len(end_activities),
            'unique_start_activities': list(start_activities),
            'unique_end_activities': list(end_activities)
        })
        
        # Calculate graph metrics
        self.model_stats.update({
            'transitions': self.process_graph.number_of_edges(),
            'avg_out_degree': sum(dict(self.process_graph.out_degree()).values()) / len(self.process_graph),
            'max_out_degree': max(dict(self.process_graph.out_degree()).values()),
            'parallel_activities': self._count_parallel_activities(),
            'cycles': len(list(nx.simple_cycles(self.process_graph)))
        })

    def _count_parallel_activities(self) -> int:
        """Count potential parallel activities based on concurrent relationships."""
        parallel_count = 0
        for event1 in self.unique_events:
            for event2 in self.unique_events:
                if event1 < event2:  # Check each pair only once
                    if self._are_parallel(event1, event2):
                        parallel_count += 1
        return parallel_count

    def _are_parallel(self, event1: str, event2: str) -> bool:
        """Check if two events can occur in parallel."""
        # Events are parallel if they appear in both orders in different traces
        order1 = order2 = False
        for trace in self.traces:
            if event1 in trace and event2 in trace:
                idx1 = trace.index(event1)
                idx2 = trace.index(event2)
                if idx1 < idx2:
                    order1 = True
                elif idx2 < idx1:
                    order2 = True
                if order1 and order2:
                    return True
        return False

    def evaluate_fitness(self) -> float:
        """Calculate fitness based on trace alignment and behavioral appropriateness."""
        trace_fitness_scores = []
        
        for trace in self.traces:
            score = self._calculate_trace_fitness(trace)
            trace_fitness_scores.append(score)
            
            if score < 0.8:  # Record issues for low-fitness traces
                self.issues['fitness'].append({
                    'trace': ' → '.join(trace),
                    'score': score,
                    'issues': self._identify_fitness_issues(trace)
                })
        
        fitness = sum(trace_fitness_scores) / len(trace_fitness_scores)
        self.metrics['fitness'] = fitness
        return fitness

    def _calculate_trace_fitness(self, trace: List[str]) -> float:
        """Calculate fitness score for a single trace."""
        score = 0.0
        max_score = 3.0  # Based on three criteria
        
        # Check start activity
        if trace[0] in self.model_stats['unique_start_activities']:
            score += 1.0
            
        # Check end activity
        if trace[-1] in self.model_stats['unique_end_activities']:
            score += 1.0
            
        # Check transitions
        valid_transitions = 0
        total_transitions = len(trace) - 1
        
        for i in range(total_transitions):
            if self.process_graph.has_edge(trace[i], trace[i + 1]):
                valid_transitions += 1
        
        if total_transitions > 0:
            score += (valid_transitions / total_transitions)
            
        return score / max_score

    def _identify_fitness_issues(self, trace: List[str]) -> List[str]:
        """Identify specific issues in a trace that affect fitness."""
        issues = []
        
        if trace[0] not in self.model_stats['unique_start_activities']:
            issues.append(f"Invalid start activity: {trace[0]}")
            
        if trace[-1] not in self.model_stats['unique_end_activities']:
            issues.append(f"Invalid end activity: {trace[-1]}")
            
        for i in range(len(trace) - 1):
            if not self.process_graph.has_edge(trace[i], trace[i + 1]):
                issues.append(f"Invalid transition: {trace[i]} → {trace[i + 1]}")
                
        return issues

    def evaluate_precision(self) -> float:
        """Calculate precision based on behavioral appropriateness."""
        # Get observed behavior
        observed_transitions = set()
        for trace in self.traces:
            for i in range(len(trace) - 1):
                observed_transitions.add((trace[i], trace[i + 1]))
        
        # Get all possible behavior from the process graph
        possible_transitions = set(self.process_graph.edges())
        
        # Calculate precision
        if not possible_transitions:
            return 0.0
            
        precision = len(observed_transitions) / len(possible_transitions)
        self.metrics['precision'] = precision
        
        # Record unused transitions
        unused = possible_transitions - observed_transitions
        for transition in unused:
            self.issues['precision'].append({
                'transition': f"{transition[0]} → {transition[1]}",
                'type': 'Unused transition'
            })
        
        return precision

    def evaluate_generalization(self) -> float:
        """Calculate generalization based on structural and behavioral characteristics."""
        # Consider multiple factors for generalization
        factors = {
            'trace_variety': self._calculate_trace_variety(),
            'structural_coverage': self._calculate_structural_coverage(),
            'behavioral_coverage': self._calculate_behavioral_coverage()
        }
        
        generalization = sum(factors.values()) / len(factors)
        self.metrics['generalization'] = generalization
        return generalization

    def _calculate_trace_variety(self) -> float:
        """Calculate variety in trace patterns."""
        unique_traces = len(set(tuple(trace) for trace in self.traces))
        return min(unique_traces / len(self.traces), 1.0)

    def _calculate_structural_coverage(self) -> float:
        """Calculate structural coverage of the process model."""
        if not self.unique_events:
            return 0.0
        
        # Calculate node coverage
        total_possible_edges = len(self.unique_events) * (len(self.unique_events) - 1)
        actual_edges = self.process_graph.number_of_edges()
        
        return actual_edges / total_possible_edges if total_possible_edges > 0 else 0.0

    def _calculate_behavioral_coverage(self) -> float:
        """Calculate behavioral coverage based on observed patterns."""
        observed_patterns = set()
        for trace in self.traces:
            # Extract event patterns (pairs and triples)
            for i in range(len(trace) - 1):
                observed_patterns.add((trace[i], trace[i + 1]))
                if i < len(trace) - 2:
                    observed_patterns.add((trace[i], trace[i + 1], trace[i + 2]))
        
        # Compare to theoretical maximum patterns
        max_patterns = (len(self.unique_events) * (len(self.unique_events) - 1) + 
                       len(self.unique_events) * (len(self.unique_events) - 1) * 
                       (len(self.unique_events) - 2))
                       
        return len(observed_patterns) / max_patterns if max_patterns > 0 else 0.0

    def generate_evaluation_report(self) -> str:
        """Generate a comprehensive evaluation report."""
        report = []
        report.append("\nProcess Model Evaluation Report")
        report.append("=" * 50)
        
        # Model Statistics
        report.append("\n1. Model Statistics")
        report.append("-" * 20)
        for key, value in self.model_stats.items():
            if isinstance(value, (int, float)):
                report.append(f"{key.replace('_', ' ').title()}: {value:.2f}" 
                            if isinstance(value, float) else f"{key.replace('_', ' ').title()}: {value}")
        
        # Conformance Metrics
        report.append("\n2. Conformance Metrics")
        report.append("-" * 20)
        for metric, value in self.metrics.items():
            report.append(f"{metric.title()}: {value:.3f}")
        
        # Analysis of Issues
        report.append("\n3. Detailed Analysis")
        report.append("-" * 20)
        
        # Fitness Issues
        if self.issues['fitness']:
            report.append("\nFitness Issues:")
            for issue in self.issues['fitness'][:5]:  # Show top 5 issues
                report.append(f"- Trace: {issue['trace']}")
                report.append(f"  Score: {issue['score']:.3f}")
                report.append(f"  Issues: {', '.join(issue['issues'])}")
        
        # Precision Issues
        if self.issues['precision']:
            report.append("\nPrecision Issues:")
            report.append(f"Found {len(self.issues['precision'])} unused transitions:")
            for issue in self.issues['precision'][:5]:
                report.append(f"- {issue['transition']}")
        
        # Generate Recommendations
        report.append("\n4. Recommendations")
        report.append("-" * 20)
        report.extend(self._generate_recommendations())
        
        return "\n".join(report)

    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations based on evaluation results."""
        recommendations = []
        
        # Fitness recommendations
        if self.metrics['fitness'] < 0.8:
            recommendations.append("- Improve process fitness:")
            if len(self.model_stats['unique_start_activities']) > 1:
                recommendations.append("  * Standardize process start activities")
            if len(self.model_stats['unique_end_activities']) > 1:
                recommendations.append("  * Standardize process end activities")
            if self.model_stats['cycles'] > 0:
                recommendations.append("  * Review and optimize process loops")
        
        # Precision recommendations
        if self.metrics['precision'] < 0.7:
            recommendations.append("- Improve process precision:")
            if len(self.issues['precision']) > 0:
                recommendations.append("  * Remove or justify unused pathways")
            if self.model_stats['parallel_activities'] > 0:
                recommendations.append("  * Review parallel activity definitions")
        
        # Generalization recommendations
        if self.metrics['generalization'] < 0.6:
            recommendations.append("- Improve process generalization:")
            if self.model_stats['unique_traces'] < self.model_stats['total_traces'] * 0.5:
                recommendations.append("  * Increase process variability")
            if self.model_stats['parallel_activities'] == 0:
                recommendations.append("  * Consider adding parallel paths where appropriate")
        
        return recommendations

if __name__ == "__main__":
    try:
        # Initialize evaluator
        print("\nInitializing Process Model Evaluator...")
        evaluator = ProcessModelEvaluator("../output/event_log_gemini.csv")
        
        # Calculate metrics
        print("\nCalculating conformance metrics...")
        fitness = evaluator.evaluate_fitness()
        precision = evaluator.evaluate_precision()
        generalization = evaluator.evaluate_generalization()
        
        # Print initial results
        print("\nEvaluation Results:")
        print("-" * 20)
        print(f"Fitness: {fitness:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Generalization: {generalization:.3f}")
        
        # Process graph statistics
        print("\nProcess Graph Statistics:")
        print("-" * 20)
        print(f"Number of activities: {len(evaluator.unique_events)}")
        print(f"Number of transitions: {evaluator.model_stats['transitions']}")
        print(f"Number of cycles: {evaluator.model_stats['cycles']}")
        print(f"Parallel activities: {evaluator.model_stats['parallel_activities']}")
        
        # Generate report
        print("\nGenerating evaluation report...")
        report = evaluator.generate_evaluation_report()
        
        # Save report with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"../output/evaluation_report_{timestamp}.txt"
        
        # Attempt to save report
        try:
            os.makedirs("output", exist_ok=True)
            with open(report_file, "w", encoding='utf-8') as f:
                f.write(report)
            print(f"\nEvaluation report saved to: {report_file}")
            
        except IOError as e:
            print(f"Error saving report to file: {e}")
            print("Printing report to console instead:")
            print("\n" + report)
            
    except FileNotFoundError:
        print("Error: Event log file not found. Please check the file path.")
    except pd.errors.EmptyDataError:
        print("Error: The event log file is empty.")
    except pd.errors.ParserError:
        print("Error: Unable to parse the event log file. Please ensure it's in the correct CSV format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())
    finally:
        print("\nProcess Model Evaluation complete.")
    