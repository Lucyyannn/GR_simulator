# GR_simulator 存储空间分配
### 决定参数-place_threshold_bytes

SSD配置中的place_threshold_bytes 是张量大小阈值。当一个张量的总字节数 ≥ 该阈值 时，该张量被分配到SSD地址区域；否则分配到DRAM地址区域。

当前配置 place_threshold_bytes = 1048576 （1MB）。

### DRAM & SSD 地址空间范围
DRAM 和 SSD的地址空间由不同的分配器管理，都采用256字节对齐。
```bash
0x000000000                DRAM区域（小张量）
    │   ← allocate_address() 递增分配
    │
    │   ...（中间未使用的地址空洞）
    │
0x800000000                SSD区域起始（大张量，≥1MB）
    │   ← allocate_address_placed() 从此递增分配
    │
    │   ...（最多到 0x107FFFFFFFF，共1TB）
    │
0x107FFFFFFFF              SSD区域结束
```
访存时，根据访存地址所在范围决定向DRAM还是SSD发送请求。

