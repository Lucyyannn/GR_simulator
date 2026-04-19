#pragma once

#include "Conv.h"

class ConvOS : public Conv {
  public:
    ConvOS(SimulationConfig config, Model* model, onnx::NodeProto& node_proto, uint32_t target_core=0);
    ConvOS(const Conv& src);
    ConvOS(SimulationConfig config, Model* model, std::string name,
           std::map<std::string, std::string>& attrs, uint32_t target_core=0);

    virtual void initialize_tiles(MappingTable& mapping_table) override;
  protected:
    virtual void initialize_instructions(Tile* tile, Mapping mapping) ;
    void init(SimulationConfig config, Model* model, onnx::NodeProto& node_proto, uint32_t target_core=0);
};