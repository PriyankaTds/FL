# server.py
import flwr as fl
from flwr.common import Metrics
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    filename="results_eps_e3.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)

# Define metric aggregation functions
def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics using weighted average."""
    # Safely extract metrics with default values
    accuracies = [m[1].get("accuracy", 0) for m in metrics]
    examples = [m[1].get("num_examples", 1) for m in metrics]  # Default to 1 if missing
    losses = [m[1].get("loss", 0) for m in metrics]
    
    # Calculate weighted averages
    total_examples = sum(examples)
    return {
        "accuracy": sum(a * e for a, e in zip(accuracies, examples)) / total_examples,
        "loss": sum(l * e for l, e in zip(losses, examples)) / total_examples,
    }

def privacy_metrics_aggregation(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Aggregate privacy metrics."""
    epsilons = [m[1].get("epsilon", 0) for m in metrics]
    return {"epsilon": sum(epsilons) / len(epsilons)} if epsilons else {"epsilon": 0.0}
def combined_metrics_aggregation(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Combine both weighted average and privacy metrics."""
    weighted_avg = weighted_average(metrics)
    privacy_metrics = privacy_metrics_aggregation(metrics)
    return {**weighted_avg, **privacy_metrics}

# Define configuration functions
def fit_config(server_round: int):
    return {
        "server_round": server_round,
        "require_num_examples": True  # Tell clients to include num_examples
    }

def evaluate_config(server_round: int):
    return {
        "server_round": server_round,
        "require_num_examples": True
    }

# Define the strategy
strategy = fl.server.strategy.FedAvg(
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
    evaluate_metrics_aggregation_fn=combined_metrics_aggregation,
    fit_metrics_aggregation_fn=combined_metrics_aggregation,
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=evaluate_config,
)

if __name__ == "__main__":
    # Start server
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

    # Process and plot results
    rounds, avg_losses, avg_accuracies, avg_epsilons = [], [], [], []

# Access metrics correctly for your Flower version
    for round_num, metrics in history.metrics_distributed.items():
        rounds.append(round_num)
        
        # Handle both fit and evaluate metrics
        if "fit" in metrics:
            avg_losses.append(metrics["fit"].get("loss", 0))
            avg_accuracies.append(metrics["fit"].get("accuracy", 0))
            avg_epsilons.append(metrics["fit"].get("epsilon", 0))
        
        if "evaluate" in metrics:
            # Use evaluate metrics if available, otherwise keep fit metrics
            avg_losses[-1] = metrics["evaluate"].get("loss", avg_losses[-1])
            avg_accuracies[-1] = metrics["evaluate"].get("accuracy", avg_accuracies[-1])
            avg_epsilons[-1] = metrics["evaluate"].get("epsilon", avg_epsilons[-1])

    # Plot the metrics after the server finishes
