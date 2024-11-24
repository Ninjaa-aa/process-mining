import pandas as pd
from collections import Counter
import itertools
from graphviz import Digraph
from tabulate import tabulate
import os
from datetime import datetime
from typing import Tuple, Set, Dict, List, FrozenSet, Any, Optional
import warnings

# Suppress pandas warnings
warnings.filterwarnings('ignore')

class PetriNetVisualizer:
    def __init__(self, output_dir: str = "output"):
        """
        Initialize PetriNetVisualizer with configurable output directory
        """
        self.dot = Digraph(comment="Petri Net")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.setup_styling()

    def setup_styling(self):
        """Configure the visual styling of the Petri net."""
        self.dot.attr(rankdir="LR")  # Left to right layout
        self.dot.attr("node", fontsize="12", fontname="Arial", width="0.6", height="0.6")
        self.dot.attr("edge", fontsize="10")

    def add_place(self, place_id: str, label: str = ""):
        """Add a place to the Petri net."""
        self.dot.node(place_id, label, shape="circle", width="0.5", height="0.5")

    def add_transition(self, trans_id: str, task_name: str):
        """Add a transition to the Petri net with both ID and task name."""
        label = f"{trans_id}\n{task_name}"
        self.dot.node(trans_id, label=label, shape="box", style="filled", 
                     color="red", fontcolor="white", width="1")

    def add_arc(self, source: str, target: str):
        """Add an arc between a place and transition."""
        self.dot.edge(source, target)

    def save(self, filename: str):
        """Save the Petri net visualization to the output directory."""
        output_path = os.path.join(self.output_dir, filename)
        try:
            file_path = self.dot.render(output_path, format="png", cleanup=True)
            print(f"Petri net visualization saved to: {file_path}")
        except Exception as e:
            print(f"Error saving Petri net visualization: {e}")


def get_sample_data() -> Tuple[List[Tuple[str, ...]], Dict[str, str]]:
    """Returns sample traces and their ID-to-task mappings to test the Alpha Algorithm."""
    sample_traces = [
        ("a", "b", "c", "d", "e", "f", "b", "d", "c", "e", "g"),
        ("a", "b", "d", "c", "e", "g"),
        ("a", "b", "c", "d", "e", "f", "b", "c", "d", "e", "f", "b", "d", "c", "e", "g"),
    ]
    id_to_event = {
        "a": "Start",
        "b": "Task B",
        "c": "Task C",
        "d": "Task D",
        "e": "Task E",
        "f": "Task F",
        "g": "End",
    }
    print("\nUsing Sample Data (Hardcoded Traces):")
    for trace in sample_traces:
        print(f"Trace: {' → '.join(trace)}")
    return sample_traces, id_to_event


def extract_event_sets_and_relationships(
    event_log_file: str, 
    use_sample_data: bool = False, 
    verbose: bool = True
) -> Tuple[Optional[Set[str]], ...]:
    """
    Implements the Alpha Algorithm with CSV support and enhanced error handling.
    """
    try:
        if use_sample_data:
            traces, id_to_event = get_sample_data()
        else:
            if verbose:
                print("\nLoading event log...")
            
            # Load CSV file instead of Excel
            df = pd.read_csv(event_log_file)
            df = df.sort_values(by=["Case ID", "Timestamp"])
            
            # Filter out missing traces
            df = df[df["Event"] != "Missing Trace"]

            # Map events to IDs
            unique_events = df["Event"].unique()
            event_to_id = {event: chr(65 + i) for i, event in enumerate(unique_events)}
            id_to_event = {v: k for k, v in event_to_id.items()}

            print("\nMapping of Tasks to IDs:")
            print(tabulate([{"ID": k, "Task": v} for k, v in id_to_event.items()], 
                         headers="keys", tablefmt="grid"))

            # Create traces with IDs
            df["Event ID"] = df["Event"].map(event_to_id)
            traces = df.groupby("Case ID")["Event ID"].apply(tuple).tolist()

            # Display trace statistics
            trace_frequencies = Counter(traces)
            if verbose:
                unique_traces = [
                    {"Trace": " → ".join(trace), "Frequency": freq} 
                    for trace, freq in trace_frequencies.items()
                ]
                print("\nTrace Statistics:")
                print(f"Total unique traces: {len(trace_frequencies)}")
                print(f"Total trace instances: {sum(trace_frequencies.values())}")
                print("\nUnique Traces and Frequencies:")
                print(tabulate(unique_traces, headers="keys", tablefmt="grid"))

        # Extract event sets
        TL = set(event for trace in traces for event in trace)
        TI = set(trace[0] for trace in traces)
        TO = set(trace[-1] for trace in traces)

        if verbose:
            print("\nEvent Sets:")
            print("\nUnique Events (TL):")
            print(tabulate([{"Event": id_to_event[e]} for e in TL], headers="keys", tablefmt="grid"))
            print("\nInitial Events (TI):")
            print(tabulate([{"Event": id_to_event[e]} for e in TI], headers="keys", tablefmt="grid"))
            print("\nFinal Events (TO):")
            print(tabulate([{"Event": id_to_event[e]} for e in TO], headers="keys", tablefmt="grid"))

        # Construct and analyze footprint matrix
        footprint_matrix = construct_footprint_matrix(traces, TL)
        if verbose:
            print("\nFootprint Matrix:")
            print(tabulate(footprint_matrix, headers="keys", tablefmt="grid"))

        # Extract relationships
        XL = {(a, b) for a, b in itertools.product(TL, TL) if footprint_matrix.at[a, b] == "→"}
        if verbose:
            print("\nCausal Relationships (XL):")
            print(tabulate([{"From": id_to_event[x[0]], "To": id_to_event[x[1]]} for x in XL], 
                         headers="keys", tablefmt="grid"))

        # Build maximal pairs
        YL = {(frozenset({a}), frozenset({b})) for a, b in XL}
        if verbose:
            print("\nMaximal Pairs (YL):")
            print(tabulate([{
                "Input Set": [id_to_event[x] for x in sorted(a)],
                "Output Set": [id_to_event[x] for x in sorted(b)]
            } for a, b in YL], headers="keys", tablefmt="grid"))

        # Construct places
        PL = {(frozenset(), frozenset(TI)), (frozenset(TO), frozenset())}.union(YL)
        if verbose:
            print("\nPlaces (PL):")
            print(tabulate([{
                "Input Set": [id_to_event[x] if x in id_to_event else "Start" for x in sorted(p[0])],
                "Output Set": [id_to_event[x] if x in id_to_event else "End" for x in sorted(p[1])]
            } for p in PL], headers="keys", tablefmt="grid"))

        # Build flow relations
        FL = set()
        for place in PL:
            input_set, output_set = place
            for t in input_set:
                FL.add((t, place))
            for t in output_set:
                FL.add((place, t))
        
        if verbose:
            print("\nFlow Relations (FL):")
            print(tabulate([{
                "Source": str(s) if isinstance(s, str) else f"Place({len(FL)})",
                "Target": str(t) if isinstance(t, str) else f"Place({len(FL)})"
            } for s, t in FL], headers="keys", tablefmt="grid"))

        return TL, TI, TO, XL, YL, PL, FL, id_to_event

    except Exception as e:
        print(f"Error during Alpha Algorithm execution: {e}")
        return None, None, None, None, None, None, None, None


def construct_footprint_matrix(traces: List[Tuple[str, ...]], TL: Set[str]) -> pd.DataFrame:
    """Construct the footprint matrix with improved relation detection."""
    events = sorted(TL)
    matrix = pd.DataFrame(index=events, columns=events, data="#")

    # Analyze direct successions
    direct_successions = set()
    for trace in traces:
        for i in range(len(trace) - 1):
            direct_successions.add((trace[i], trace[i + 1]))

    # Fill matrix
    for a, b in itertools.product(events, repeat=2):
        if a == b:
            matrix.at[a, b] = "#"  # Self-relations
        else:
            forward = (a, b) in direct_successions
            backward = (b, a) in direct_successions
            
            if forward and not backward:
                matrix.at[a, b] = "→"
            elif backward and not forward:
                matrix.at[a, b] = "←"
            elif forward and backward:
                matrix.at[a, b] = "||"
            else:
                matrix.at[a, b] = "#"

    return matrix


def build_and_visualize_petri_net(
    TL: Set[str],
    TI: Set[str],
    TO: Set[str],
    XL: Set[Tuple[str, str]],
    PL: Set[Tuple[FrozenSet[str], FrozenSet[str]]],
    FL: Set[Tuple[Any, Any]],
    id_to_event: Dict[str, str],
    output_file: str = "petri_net"
) -> None:
    """Constructs and visualizes the Petri Net with enhanced place labeling."""
    try:
        visualizer = PetriNetVisualizer()

        # Create meaningful place labels
        place_labels = {}
        for i, place in enumerate(PL, 1):
            if not place[0] and place[1] == frozenset(TI):
                place_labels[place] = "Start"
            elif not place[1] and place[0] == frozenset(TO):
                place_labels[place] = "End"
            else:
                input_set = "_".join(sorted(id_to_event[t] for t in place[0])) if place[0] else "Start"
                output_set = "_".join(sorted(id_to_event[t] for t in place[1])) if place[1] else "End"
                place_labels[place] = f"P{i}_{input_set[:10]}_{output_set[:10]}"

        # Add transitions
        for transition in TL:
            visualizer.add_transition(transition, id_to_event[transition])

        # Add places
        for place, label in place_labels.items():
            visualizer.add_place(label)

        # Add arcs
        for source, target in FL:
            source_label = source if isinstance(source, str) else place_labels.get(source, "Unknown")
            target_label = target if isinstance(target, str) else place_labels.get(target, "Unknown")
            visualizer.add_arc(source_label, target_label)

        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{output_file}_{timestamp}"
        visualizer.save(output_file)

    except Exception as e:
        print(f"Error building Petri net visualization: {e}")


if __name__ == "__main__":
    try:
        # Configuration
        use_sample_data = False
        event_log_file = "output/event_log_gemini.csv"  # Update path to match your CSV file
        
        print("\nProcess Mining - Alpha Algorithm Analysis")
        print("=" * 50)
        
        # Extract and analyze
        result = extract_event_sets_and_relationships(
            event_log_file, 
            use_sample_data=use_sample_data
        )
        
        if all(result):
            TL, TI, TO, XL, YL, PL, FL, id_to_event = result
            build_and_visualize_petri_net(TL, TI, TO, XL, PL, FL, id_to_event)
        else:
            print("\nError: Could not complete Alpha Algorithm analysis.")
            
    except Exception as e:
        print(f"\nError in main execution: {e}")