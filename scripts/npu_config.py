#!/usr/bin/env python3
"""Materialize one area/power-constrained NPU compute reconfiguration.

The baseline JSON remains immutable.  A generated runtime base config is
written to the requested output path from explicit Cube/Vector CLI inputs.
One simulator core represents ``cube_compression`` physical Cube cores; its
Cube height and Vector width preserve aggregate throughput as closely as the
simulator's integer fields permit.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path


REFERENCE_VECTOR_WIDTH_BITS = 2048
REFERENCE_VECTOR_GFLOPS = 256.0
REFERENCE_CUBE_TFLOPS = 8.0
# The simulator's baseline Cube shape is 64x256 at 1 GHz.  Use its exact
# aggregate rate when materializing DSE points so the physical baseline point
# reproduces the original baseline throughput instead of the rounded 8 TFLOPS
# area/power reference.
SIMULATOR_CUBE_TFLOPS_AT_1GHZ = 8.192
CUBE_AREA_MM2 = 2.57
CUBE_POWER_W = 3.13
VECTOR_AREA_MM2 = 0.70
VECTOR_POWER_W = 0.46


@dataclass(frozen=True)
class DesignPoint:
    cube_cores: int
    vector_cores: int
    vector_width_bits: int


@dataclass(frozen=True)
class ResourceUsage:
    area_mm2: float
    power_w: float


def resource_usage(point: DesignPoint) -> ResourceUsage:
    vector_scale = point.vector_width_bits / REFERENCE_VECTOR_WIDTH_BITS
    return ResourceUsage(
        area_mm2=(
            point.cube_cores * CUBE_AREA_MM2
            + point.vector_cores * VECTOR_AREA_MM2 * vector_scale
        ),
        power_w=(
            point.cube_cores * CUBE_POWER_W
            + point.vector_cores * VECTOR_POWER_W * vector_scale
        ),
    )


def baseline_design(config: dict, cube_compression: int = 4) -> DesignPoint:
    physical = config.get("metadata", {}).get("physical_compute_units")
    if physical is not None:
        return DesignPoint(
            cube_cores=int(physical["cube_cores"]),
            vector_cores=int(physical["vector_cores"]),
            vector_width_bits=int(physical["vector_width_bits"]),
        )
    return DesignPoint(
        cube_cores=int(config["num_cores"]) * cube_compression,
        vector_cores=int(config["num_cores"]) * cube_compression,
        vector_width_bits=REFERENCE_VECTOR_WIDTH_BITS,
    )


def target_compute(point: DesignPoint, frequency_mhz: float) -> dict[str, float]:
    """Return exact architectural throughputs used by the cost model.

    Vector FLOP/s includes the simulator's two-FLOP convention.  The paper
    Vector-op denominator is half of that value.
    """

    frequency_scale = frequency_mhz / 1000.0
    vector_flops = (
        point.vector_cores
        * REFERENCE_VECTOR_GFLOPS
        * 1e9
        * (point.vector_width_bits / REFERENCE_VECTOR_WIDTH_BITS)
        * frequency_scale
    )
    return {
        "cube_flops": (
            point.cube_cores
            * SIMULATOR_CUBE_TFLOPS_AT_1GHZ
            * 1e12
            * frequency_scale
        ),
        "vector_flops": vector_flops,
        "vector_ops": vector_flops / 2.0,
    }


def round_to_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(math.floor(value / multiple + 0.5)) * multiple)


def materialize_config(
    baseline: dict,
    point: DesignPoint,
    *,
    cube_compression: int = 4,
) -> tuple[dict, dict]:
    if min(point.cube_cores, point.vector_cores, point.vector_width_bits) <= 0:
        raise ValueError("Cube cores, Vector cores, and Vector width must be positive")
    if cube_compression <= 0:
        raise ValueError("cube_compression must be positive")

    output = copy.deepcopy(baseline)
    first_core = baseline["core_config"]["core_0"]
    frequency_mhz = float(baseline["core_freq"])
    frequency_hz = frequency_mhz * 1e6
    precision_bytes = int(baseline["precision"])
    cube_width = int(first_core["core_width"])
    simulator_cores = int(math.ceil(point.cube_cores / cube_compression))
    target = target_compute(point, frequency_mhz)

    ideal_height = target["cube_flops"] / (
        2.0 * simulator_cores * cube_width * frequency_hz
    )
    cube_height = max(1, int(math.floor(ideal_height + 0.5)))
    ideal_vector_bits = (
        point.vector_cores * point.vector_width_bits / simulator_cores
    )
    vector_bits = round_to_multiple(ideal_vector_bits, 8)

    core_config = {}
    for index in range(simulator_cores):
        core = copy.deepcopy(first_core)
        core["core_width"] = cube_width
        core["core_height"] = cube_height
        core["vector_process_bit"] = vector_bits
        core_config[f"core_{index}"] = core
    output["num_cores"] = simulator_cores
    output["core_config"] = core_config

    actual_cube = (
        simulator_cores * cube_width * cube_height * 2.0 * frequency_hz
    )
    actual_vector_flops = (
        simulator_cores
        * (vector_bits / (8.0 * precision_bytes))
        * 2.0
        * frequency_hz
    )
    mapping = {
        "physical_design": {
            "cube_cores": point.cube_cores,
            "vector_cores": point.vector_cores,
            "vector_width_bits": point.vector_width_bits,
        },
        "resource_usage": resource_usage(point).__dict__,
        "cube_compression": cube_compression,
        "simulator_cores": simulator_cores,
        "simulator_cube_width": cube_width,
        "ideal_simulator_cube_height": ideal_height,
        "simulator_cube_height": cube_height,
        "ideal_simulator_vector_width_bits": ideal_vector_bits,
        "simulator_vector_width_bits": vector_bits,
        "target_cube_flops": target["cube_flops"],
        "actual_cube_flops": actual_cube,
        "cube_relative_error": actual_cube / target["cube_flops"] - 1.0,
        "target_vector_flops": target["vector_flops"],
        "actual_vector_flops": actual_vector_flops,
        "vector_relative_error": actual_vector_flops / target["vector_flops"] - 1.0,
    }
    output.setdefault("metadata", {})["npu_reconfiguration"] = mapping
    output["metadata"]["derived_fp16_tflops"] = actual_cube / 1e12
    output["metadata"]["derived_vector_tflops"] = actual_vector_flops / 1e12
    return output, mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cube-cores", type=int, required=True)
    parser.add_argument("--vector-cores", type=int, required=True)
    parser.add_argument("--vector-width", type=int, required=True)
    parser.add_argument("--cube-compression", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    point = DesignPoint(args.cube_cores, args.vector_cores, args.vector_width)
    output, mapping = materialize_config(
        baseline, point, cube_compression=args.cube_compression
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(mapping, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
