本工具用于处理 MODIS 热红外数据，支持以下数据的自动化处理：

热红外波段：（27、28、29、30、31、32 ）波段（波段可自定义）

发射率数据：31、32 波段发射率

地表温度数据：LST（Land Surface Temperature）

质量控制标识：QC（Quality Control）

处理后文件包含以下列数据：
经度 纬度 TIR27 TIR28 TIR29 TIR30 TIR31 TIR32 发射率31 发射率32 LST温度 QC标识

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
- 日期目录：8位数字（示例：230806 表示 2023年8月6日）
- HDF文件：必须包含相同的时间戳标识（示例中`.h08v03.061.2023086233058`）
- 匹配规则：LST和TIR文件时间戳需完全一致

---

## 🛠️ 环境配置

### 依赖安装
```bash
# 基础依赖
pip install numpy pandas pyyaml scipy tqdm

# GDAL推荐通过conda安装
conda install -c conda-forge gdal
```

### 环境验证
```bash
# 检查GDAL版本
gdalinfo --version
# 应输出类似 GDAL 3.6.4 的版本信息
```

---

## 🚀 快速开始

### 单日数据处理
```python
from your_module import 猫迪大祭司

# 处理指定日期数据
猫迪大祭司().core(
    base_path='datatest',  # 数据根目录
    date='230806'          # 日期目录名
)
```

### 批量处理模式
```python
# 自动处理根目录下所有日期
猫迪大祭司().batch_core('datatest')
```

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
│   └── night/
├── 230807/
└── merged/
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
   - 自动识别同时间戳的LST和TIR文件
   - 创建处理目录（`proc_`前缀）

2. **数据投影**  
   - 将HDF转换为ENVI格式（WGS84坐标系）
   - 分辨率自动计算（约1km）

3. **格式转换**  
   - 生成ASCII中间文件
   - 过滤无效数据（QC=0的有效数据）

4. **数据合并**  
   - 多日数据智能合并
   - 坐标重复时自动取平均值

5. **亮温转换**  
   - 使用Planck公式计算亮温：  
     ![BT Formula](https://latex.codecogs.com/png.latex?T%3D%5Cfrac%7Bhc%7D%7Bk%5Clambda%5Cln%5Cleft%28%5Cfrac%7B2hc%5E2%7D%7B%5Clambda%5E5L%7D&plus;1%5Cright%29%7D)

---

## 🚨 常见问题

### Q1: 文件匹配失败
**现象**：日志中显示`No matching files`  
✅ 解决方案：  
1. 检查文件名时间戳是否一致  
2. 验证文件路径是否符合规范  
3. 确保HDF文件未被损坏（可用GDAL打开验证）

### Q2: GDAL初始化错误
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
- 系统默认保留2位小数（经度纬度）
- 可通过修改源码调整精度：  
```python
# 在 ascii_to_column 方法中修改
lon_lat_fmt = ['%.2f', '%.2f']  # 改为 %.3f 可提高精度
```

---

## ⚙️ 高级配置

### 修改投影参数
```python
# 在 save_to_raster 方法中修改
target_srs = osr.SpatialReference()
target_srs.ImportFromEPSG(3857)  # 改为Web墨卡托投影
```

### 调整亮温计算
```python
# 在 radiometric_to_bt 方法中修改
modis_wavelengths = np.array([...])  # 自定义波段参数
emis_calibration = 0.002 * ... + 0.49  # 发射率计算公式
```

---

## 📜 注意事项
1. 路径中请避免使用中文和空格
2. 建议使用SSD硬盘处理大数据量
3. 夜间数据处理耗时约为白天的1.5倍
4. 最终输出文件编码为UTF-8

---

> 📧 如有其他问题，请联系：your_email@example.com  
> 🔗 项目地址：[github.com/yourname/modis-processor](https://github.com/yourname/modis-processor)