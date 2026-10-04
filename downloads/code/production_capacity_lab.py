"""Simulation-based capacity planning for a parallel-machine production cell.

Run from the root directory of the LaTeX project with

    python3 code/production_capacity_lab.py

The implementation uses only the Python standard library.
"""

from dataclasses import dataclass
from math import log, sqrt
from random import Random
from statistics import mean, stdev


@dataclass(frozen=True)
class ModelParameters:
    horizon: float = 480.0          # minutes in one production shift
    arrival_rate: float = 0.11      # expected arrivals per minute
    mean_service: float = 18.0      # mean processing time in minutes
    service_sigma: float = 0.45     # lognormal shape parameter
    turnaround_target: float = 45.0
    machine_cost: float = 220.0
    waiting_cost: float = 0.70
    late_job_cost: float = 30.0


@dataclass(frozen=True)
class SimulationResult:
    jobs: int
    mean_wait: float
    utilization: float
    late_jobs: int
    total_cost: float


def generate_sample_path(parameters, seed):
    """Generate arrivals and customer-specific processing requirements."""
    rng = Random(seed)
    arrivals = []
    services = []
    arrival_time = 0.0
    mu = log(parameters.mean_service) - 0.5 * parameters.service_sigma ** 2

    while True:
        arrival_time += rng.expovariate(parameters.arrival_rate)
        if arrival_time > parameters.horizon:
            break
        arrivals.append(arrival_time)
        services.append(rng.lognormvariate(mu, parameters.service_sigma))

    return arrivals, services


def simulate_production_cell(num_machines, sample_path, parameters):
    """Run an FCFS discrete-event simulation on a fixed sample path."""
    arrivals, services = sample_path
    available_at = [0.0] * num_machines
    total_wait = 0.0
    busy_time_in_shift = 0.0
    late_jobs = 0

    for arrival, service in zip(arrivals, services):
        machine = min(range(num_machines), key=available_at.__getitem__)
        start = max(arrival, available_at[machine])
        finish = start + service
        wait = start - arrival

        total_wait += wait
        if finish - arrival > parameters.turnaround_target:
            late_jobs += 1
        busy_time_in_shift += max(
            0.0,
            min(finish, parameters.horizon) - start,
        )
        available_at[machine] = finish

    jobs = len(arrivals)
    mean_wait = total_wait / jobs if jobs else 0.0
    utilization = busy_time_in_shift / (
        num_machines * parameters.horizon
    )
    total_cost = (
        parameters.machine_cost * num_machines
        + parameters.waiting_cost * total_wait
        + parameters.late_job_cost * late_jobs
    )
    return SimulationResult(
        jobs=jobs,
        mean_wait=mean_wait,
        utilization=utilization,
        late_jobs=late_jobs,
        total_cost=total_cost,
    )


def simulate_alternatives(
    alternatives,
    replications,
    parameters,
    seed,
    use_common_random_numbers=True,
):
    """Simulate all machine counts with common or independent random numbers."""
    results = {machines: [] for machines in alternatives}
    for replication in range(replications):
        if use_common_random_numbers:
            path = generate_sample_path(parameters, seed + replication)
            for machines in alternatives:
                results[machines].append(
                    simulate_production_cell(machines, path, parameters)
                )
        else:
            for machines in alternatives:
                path_seed = seed + 100000 * machines + replication
                path = generate_sample_path(parameters, path_seed)
                results[machines].append(
                    simulate_production_cell(machines, path, parameters)
                )
    return results


def confidence_interval(values, confidence_multiplier=1.96):
    estimate = mean(values)
    half_width = confidence_multiplier * stdev(values) / sqrt(len(values))
    return estimate, half_width


def summarize(results):
    rows = []
    for machines, replications in sorted(results.items()):
        costs = [result.total_cost for result in replications]
        estimate, half_width = confidence_interval(costs)
        rows.append(
            (
                machines,
                mean(result.mean_wait for result in replications),
                mean(result.utilization for result in replications),
                mean(result.late_jobs for result in replications),
                estimate,
                half_width,
            )
        )
    return rows


def paired_difference_interval(results, first, second):
    """CRN interval for E[cost(first) - cost(second)]."""
    differences = [
        left.total_cost - right.total_cost
        for left, right in zip(results[first], results[second])
    ]
    return confidence_interval(differences)


def independent_difference_interval(results, first, second):
    """Independent-sample interval for E[cost(first) - cost(second)]."""
    first_costs = [result.total_cost for result in results[first]]
    second_costs = [result.total_cost for result in results[second]]
    estimate = mean(first_costs) - mean(second_costs)
    standard_error = sqrt(
        stdev(first_costs) ** 2 / len(first_costs)
        + stdev(second_costs) ** 2 / len(second_costs)
    )
    return estimate, 1.96 * standard_error


def print_summary(rows):
    print("Machines  Mean wait  Utilization  Late jobs  Mean cost     95% CI")
    for machines, wait, utilization, late, cost, half_width in rows:
        print(
            f"{machines:8d}  {wait:9.2f}  {utilization:11.3f}"
            f"  {late:9.2f}  {cost:9.2f}  +/- {half_width:6.2f}"
        )


def main():
    parameters = ModelParameters()
    alternatives = range(2, 7)

    # Stage 1: use CRN to compare all candidate capacity levels.
    selection_results = simulate_alternatives(
        alternatives,
        replications=1000,
        parameters=parameters,
        seed=100,
        use_common_random_numbers=True,
    )
    selection_summary = summarize(selection_results)
    print("Selection experiment")
    print_summary(selection_summary)
    selected = min(selection_summary, key=lambda row: row[4])[0]
    print(f"\nSelected capacity: {selected} machines")

    # Stage 2: validate the selected design on fresh sample paths.
    validation_results = simulate_alternatives(
        [selected],
        replications=5000,
        parameters=parameters,
        seed=10000,
        use_common_random_numbers=True,
    )
    validation_costs = [
        result.total_cost for result in validation_results[selected]
    ]
    validation_mean, validation_half_width = confidence_interval(
        validation_costs
    )
    print(
        f"Validation cost: {validation_mean:.2f}"
        f" +/- {validation_half_width:.2f}"
    )

    # Demonstrate the precision gained from pairing sample paths.
    crn_difference, crn_half_width = paired_difference_interval(
        selection_results, 3, 4
    )
    independent_results = simulate_alternatives(
        [3, 4],
        replications=1000,
        parameters=parameters,
        seed=20000,
        use_common_random_numbers=False,
    )
    ind_difference, ind_half_width = independent_difference_interval(
        independent_results, 3, 4
    )
    print("\nCost difference, 3 machines minus 4 machines")
    print(f"CRN:         {crn_difference:7.2f} +/- {crn_half_width:.2f}")
    print(f"Independent: {ind_difference:7.2f} +/- {ind_half_width:.2f}")


if __name__ == "__main__":
    main()
