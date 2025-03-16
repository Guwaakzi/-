---

# MODIS 热红外数据自动化处理工具

本工具用于自动化处理 MODIS 热红外数据，支持热红外波段、发射率数据、地表温度数据（LST）和质量控制标识（QC）的提取与处理。工具支持自定义波段配置，并生成包含经度、纬度、热红外亮温、发射率、LST 温度和 QC 标识的处理结果。

---

## 📂 输入数据结构要求

### 目录结构
```
数据根目录（如：datatest/）
├── 日期目录（如：230806）
│   ├── day/
│   │   ├── LST/
│   │   │   └── *.hdf（示例：MOD11A1.A20230806.h08v03.061.2023086233058.hdf）
│   │   └── TIR/
│   │       └── *.hdf（示例：MOD021KM.A2023086.h08v03.061.2023086233058.hdf）
│   └── night/
│       ├── LST/
│       └── TIR/
└── 其他日期目录...
```

### 文件命名规范
- **日期目录**：8位数字（示例：`230806` 表示 2023年8月6日）
- **HDF文件**：必须包含相同的时间戳标识（示例中 `.h08v03.061.2023086233058`）
- **匹配规则**：LST 和 TIR 文件的时间戳需完全一致

---

## 🛠️ 环境配置

### 依赖安装
```bash
# 基础依赖
pip install -r requirements.txt

# GDAL推荐通过conda安装
conda install -c conda-forge gdal
```

### 环境验证
```bash
# 检查GDAL版本
gdalinfo --version
# 应输出类似 GDAL 3.8.4 的版本信息
```

---

## 🚀 快速开始

### 批量处理模式
```python
# 自动处理根目录下所有日期
猫迪大祭司().batch_core('datatest')
```

### 代码运行记录（YAML 文件）
```yaml
day:
  t0230:
    append_data: true         # 是否追加数据到文件
    ascii_to_column: true     # 是否转换为列数据格式
    load_modis_data: true     # 是否加载 MODIS 数据
    save_to_ascii: true       # 是否保存为 ASCII 格式
    save_to_raster: true      # 是否保存为 ENVI 格式
```

#### 防中断机制
如果代码运行中断，某天日期下 `append_data` 为 `true`，则在重新运行代码时，会自动跳过该天的数据处理流程，避免重复计算，节省时间。

---

## 📂 输出结构

### 生成目录结构
```
proc_datatest/
├── 230806/
│   ├── day/
│   │   └── 时间戳目录（如：0820）
│   │       ├── 0820.dat        # ENVI格式数据文件
│   │       ├── 0820.hdr        # ENVI头文件
│   │       └── 0820.txt        # 中间文本数据
│   ├── night/
│   ├── daycol.txt         # 当天白天数据
│   ├── daycol.yaml        # 当天白天数据处理信息
│   ├── nightcol.txt       # 当天夜间数据
│   └── nightcol.yaml      # 当天夜晚数据处理信息
├── 230807/
├── L_daycol.txt           # 合并后的白天原始数据
├── L_nightcol.txt         # 合并后的夜晚原始数据
├── bt_L_daycol.txt        # 最终白天亮温数据
└── bt_L_nightcol.txt      # 最终夜晚亮温数据
```

### 数据文件说明
| 文件名 | 格式 | 内容结构 |
|-------|------|---------|
| `bt_L_*col.txt` | 制表符分隔 | `经度 纬度 TIR27...TIR32亮温 发射率31 发射率32 LST温度 QC标识` |
| `L_*col.txt` | 制表符分隔 | 中间数据，包含原始辐射值 |

---

## 🔄 处理流程
1. **文件匹配**  
   - 自动识别同时间戳的 LST 和 TIR 文件
   - 创建处理目录（`proc_`前缀）

2. **数据投影**  
   - 将 HDF 转换为 ENVI 格式（WGS84 坐标系）
   - 分辨率自动计算（约 1km）

3. **格式转换**  
   - 生成 ASCII 中间文件
   - 过滤无效数据（QC=0 的有效数据）

4. **数据合并**  
   - 多日数据智能合并
   - 坐标重复时自动取平均值

5. **亮温转换**  
   - 使用 Planck 公式计算亮温：  
     ![BT Formula](https://latex.codecogs.com/png.latex?T%3D%5Cfrac%7Bhc%7D%7Bk%5Clambda%5Cln%5Cleft%28%5Cfrac%7B2hc%5E2%7D%7B%5Clambda%5E5L%7D&plus;1%5Cright%29%7D)

---

## ⚙️ 配置文件说明 (`config/config.yaml`)

### 常量参数
```yaml
constants:
  planck_constant: 6.62607015e-34  # 普朗克常数 h, J·s
  speed_of_light: 2.99792458e8     # 光速 c, m/s
  boltzmann_constant: 1.380649e-23 # 玻尔兹曼常数 k, J/K
```

### 热红外波段配置
```yaml
TIR_bands:
  band_numbers: [7, 8, 9, 10, 11, 12]  # Python 中的索引从 0 开始
  band_names: ['TIR_27', 'TIR_28', 'TIR_29', 'TIR_30', 'TIR_31', 'TIR_32']
```

### MODIS 参数
```yaml
modis_parameters:
  wavelengths: [6.715e-6, 7.325e-6, 8.550e-6, 9.730e-6, 11.030e-6, 12.020e-6]  # 波段 27-32 的中心波长 (单位: 米)
  radiance_scales: [0.000117557, 0.00019245, 0.000532487, 0.000406323, 0.000840022, 0.000729698]  # 波段 27-32 的辐射尺度
  radiance_offsets: [2730.5833, 2317.4883, 2730.5835, 1560.3333, 1577.3397, 1658.2213]  # 波段 27-32 的辐射偏移
  emissivity_coefficients:  # 发射率计算参数
    slope: 0.002  # 发射率计算斜率
    intercept: 0.49  # 发射率计算截距
  lst_coefficients:  # 地表温度计算参数
    slope: 0.02  # 地表温度计算斜率
```

---

## 🚨 常见问题

### Q1: 文件匹配失败
**现象**：日志中显示 `No matching files`  
✅ 解决方案：  
1. 检查文件名时间戳是否一致  
2. 验证文件路径是否符合规范  
3. 确保 HDF 文件未被损坏（可用 GDAL 打开验证）

### Q2: GDAL 初始化错误
**现象**：`RuntimeError: Unable to open dataset`  
✅ 解决方案：  
```bash
# Linux/Mac
export GDAL_DATA=/path/to/anaconda3/share/gdal

# Windows
set GDAL_DATA=C:\path\to\anaconda3\Library\share\gdal
```

### Q3: 坐标精度问题
**现象**：合并后数据量异常减少  
✅ 处理逻辑：  
- 系统默认保留 2 位小数（经度纬度）
- 可通过修改源码调整精度：  
```python
# 在 ascii_to_column 方法中修改
lon_lat_fmt = ['%.2f', '%.2f']  # 改为 %.3f 可提高精度
```

---

## 📜 注意事项
1. 路径中请避免使用中文和空格
2. 建议使用 SSD 硬盘处理大数据量
3. 夜间数据处理耗时约为白天的 1.5 倍
4. 最终输出文件编码为 UTF-8

---

> 📧 如有其他问题，请联系：guwaakzi@gmail.com  
> 🔗 项目地址：[https://github.com/Guwaakzi/MODIS_Automated_Processing]

---

### 版本记录
- **v1.0.0**：初始版本，支持 MODIS 热红外数据自动化处理
- **v1.1.0**：新增批量处理模式，优化文件匹配逻辑

---