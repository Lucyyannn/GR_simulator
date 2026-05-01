#ifndef HBM_H
#define HBM_H

#include "Dram.h"

class Hbm : public Ramulator2Memory {
 public:
  explicit Hbm(const SimulationConfig& config);
};

#endif
