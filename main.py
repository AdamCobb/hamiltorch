import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True,
                        choices=["sample_size", "sample_size_analytic", "snn_ablation",
                                 "rmhmc", "gp", "symmetry_control", "pair_mode", "traj_length", "plot_ess"],
                        help="Which experiment or plot to run")
    parser.add_argument("--device", default="cpu", help="Compute device (cpu, cuda, mps:0)")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny chains/epochs to validate the pipeline end-to-end")
    args = parser.parse_args()

    # Must be set before hamiltorch is imported: smoke mode is read at import time.
    if args.smoke:
        os.environ["HAMILTORCH_SMOKE"] = "1"

    import torch
    torch.set_default_device(args.device)

    from hamiltorch import (
        surrogate_neural_ode_hmc_sample_size_experiment,
        surrogate_neural_ode_hmc_sample_size_experiment_analytic,
        snn_gradient_ablation_experiment,
        rmhmc_experiment,
        gp_sample_size_experiment,
        symmetrization_control_experiment,
        pair_mode_experiment,
        trajectory_length_experiment,
    )
    from hamiltorch.plot_utils import plot_ess_vs_training_size

    if args.experiment == "sample_size":
        surrogate_neural_ode_hmc_sample_size_experiment(args.device)
    elif args.experiment == "sample_size_analytic":
        surrogate_neural_ode_hmc_sample_size_experiment_analytic()
    elif args.experiment == "snn_ablation":
        snn_gradient_ablation_experiment(args.device)
    elif args.experiment == "rmhmc":
        rmhmc_experiment(args.device)
    elif args.experiment == "gp":
        gp_sample_size_experiment(args.device)
    elif args.experiment == "symmetry_control":
        symmetrization_control_experiment(args.device)
    elif args.experiment == "pair_mode":
        pair_mode_experiment(args.device)
    elif args.experiment == "traj_length":
        trajectory_length_experiment(args.device)
    elif args.experiment == "plot_ess":
        plot_ess_vs_training_size(
            csv_path="experiments/diagnostic_results.csv",
            output_path="experiments/ess_vs_training_size.png",
        )
        print("ESS vs training size figure saved.")
