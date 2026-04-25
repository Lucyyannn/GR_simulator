#include "Hbm.h"

Hbm::Hbm(const SimulationConfig& config)
    : Ramulator2Memory(config, config.hbm, "HBM") {}
