import os
import yaml
import datetime
from scipy.ndimage import zoom
from osgeo import gdal, osr
import numpy as np
import tempfile
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
from hijack import blinker, EasyDict_gu, ryaml, dyaml, 计时器
gdal.UseExceptions() 

from collections import defaultdict

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class 猫迪大祭司:
    def __init__(self):
        self.band_data = {}
        self.geo_index = {}
        self.Matches = None
        self.ascll = None
        self.base_name = None
        self.info = None
        self.yamls = None
        self.config = ryaml('config/config.yaml')

    def find_matching_files(self, base_path, date):
        """
        在给定的日期文件夹中找到对应的LST和TIR文件，并创建保存匹配文件的文件夹。
        """
        day_path = os.path.join(base_path, date, 'day')
        night_path = os.path.join(base_path, date, 'night')
        save_base_path = os.path.join('proc_' + base_path,  date)

        # 创建字典来存储文件名中的时间到文件路径的映射
        day_lst_files = defaultdict(list)
        day_tir_files = defaultdict(list)
        night_lst_files = defaultdict(list)
        night_tir_files = defaultdict(list)

        # 遍历day文件夹下的LST和TIR文件
        for folder_name in ['LST', 'TIR']:
            folder_path = os.path.join(day_path, folder_name)
            for file in os.listdir(folder_path):
                time_stamp = file.split('.')[2]  # 获取时间戳
                if folder_name == 'LST':
                    day_lst_files[time_stamp].append(os.path.join(folder_path, file))
                else:
                    day_tir_files[time_stamp].append(os.path.join(folder_path, file))

        # 遍历night文件夹下的LST和TIR文件
        for folder_name in ['LST', 'TIR']:
            folder_path = os.path.join(night_path, folder_name)
            for file in os.listdir(folder_path):
                time_stamp = file.split('.')[2]  # 获取时间戳
                if folder_name == 'LST':
                    night_lst_files[time_stamp].append(os.path.join(folder_path, file))
                else:
                    night_tir_files[time_stamp].append(os.path.join(folder_path, file))

        # 匹配day中的LST和TIR文件，并创建保存文件夹
        day_matches = []
        for time_stamp in day_lst_files:
            if time_stamp in day_tir_files:
                for lst, tir in zip(day_lst_files[time_stamp], day_tir_files[time_stamp]):
                    # 创建保存文件夹路径
                    day_save_folder = os.path.join(save_base_path, 'day', time_stamp)
                    os.makedirs(day_save_folder, exist_ok=True)  # 创建文件夹

                    # 将保存文件夹路径添加到匹配项中
                    day_matches.append((time_stamp, tir, lst, day_save_folder))

        # 匹配night中的LST和TIR文件，并创建保存文件夹
        night_matches = []
        for time_stamp in night_lst_files:
            if time_stamp in night_tir_files:
                for lst, tir in zip(night_lst_files[time_stamp], night_tir_files[time_stamp]):
                    # 创建保存文件夹路径
                    night_save_folder = os.path.join(save_base_path, 'night', time_stamp)
                    os.makedirs(night_save_folder, exist_ok=True)  # 创建文件夹

                    # 将保存文件夹路径添加到匹配项中
                    night_matches.append((time_stamp, tir, lst, night_save_folder))

        self.Matches = [ 
            ['day', day_matches, os.path.join(save_base_path, 'daycol.txt'), os.path.join(save_base_path, 'daycol.yaml')], 
            ['night', night_matches, os.path.join(save_base_path, 'nightcol.txt'), os.path.join(save_base_path, 'nightcol.yaml')] 
        ]

    def load_modis_data(self, match):
        def load_message(match):
            dataset = gdal.Open(match)
            sub_datasets = dataset.GetSubDatasets()
            print("Available Subdatasets:")
            for i, subdataset in enumerate(sub_datasets):
                print(f"{i + 1}: {subdataset[0]}")
        def read_band(dataset, band_numbers=None):
            if band_numbers is None:
                band = dataset.GetRasterBand(1)
                data = band.ReadAsArray()
            else:
                data = {}
                for band_number in [7, 8, 9, 10, 11, 12]:
                    band = TIR_dataset.GetRasterBand(band_number)
                    band_data = band.ReadAsArray()
                    data[band_number] = band_data
            return data

        #band_data = {}
        TIR_path = match[1]
        LST_path = match[2]
        # 打开主数据集 (EV_1KM_Emissive)
        TIR_dataset = gdal.Open(f'HDF4_EOS:EOS_SWATH:"{TIR_path}":MODIS_SWATH_Type_L1B:EV_1KM_Emissive')
        LST_dataset = gdal.Open(f'HDF4_EOS:EOS_SWATH:"{LST_path}":MOD_Swath_LST:LST')
        QC_dataset = gdal.Open(f'HDF4_EOS:EOS_SWATH:"{LST_path}":MOD_Swath_LST:QC')
        Emis31_dataset = gdal.Open(f'HDF4_EOS:EOS_SWATH:"{LST_path}":MOD_Swath_LST:Emis_31')
        Emis32_dataset = gdal.Open(f'HDF4_EOS:EOS_SWATH:"{LST_path}":MOD_Swath_LST:Emis_32')
        longitude_dataset = gdal.Open(f'HDF4_EOS:EOS_SWATH:"{LST_path}":MOD_Swath_LST:Longitude')
        latitude_dataset = gdal.Open(f'HDF4_EOS:EOS_SWATH:"{LST_path}":MOD_Swath_LST:Latitude')


        # TIR_datas = read_band(TIR_dataset, band_numbers=[7, 8, 9, 10, 11, 12])
        # LST_data = read_band(LST_dataset)
        # QC_data = read_band(QC_dataset)
        # Emis31_data = read_band(Emis31_dataset)
        # Emis32_data = read_band(Emis32_dataset)
        # longitude_data = read_band(longitude_dataset)
        # latitude_data = read_band(latitude_dataset)


        # for label, band in {'TIR': TIR_datas, 'Emis31': Emis31_data, 'Emis32': Emis32_data,
        #  'LST': LST_data, 'QC': QC_data, 'lon':longitude_data, 'lat':latitude_data}.items():
        #     if label == 'TIR':
        #         for band_number, data in band.items():
        #             band_data[f"{label}_{band_number + 20}"] = data
        #     else:
        #         band_data[label] = band

        # # 检查并打印结果
        # print("\n加载的所有波段数据:")
        # for label, data in band_data.items():
        #     if data is not None:
        #         print(f"{label} - Shape: {data.shape}")
        #         print(f"{label} - Min: {data.min()}, Max: {data.max()}")
        #     else:
        #         print(f"{label} - 数据加载失败")

        # # 关闭数据集以释放资源
        # TIR_dataset = None
        # LST_dataset = None
        # QC_dataset = None
        # Emis31_dataset = None
        # Emis32_dataset = None
        # longitude_dataset = None
        # latitude_dataset = None
        self.band_data = {
            'TIR': TIR_dataset,
            'Emis31': Emis31_dataset,
            'Emis32': Emis32_dataset,
            'LST': LST_dataset,
            'QC': QC_dataset,
            'lon': longitude_dataset,
            'lat': latitude_dataset
        }
        self.yamls([self.info, 'load_modis_data'], True)

    def generate_lat_lon(self, hdr_path):
        """
        从ENVI头文件中提取经度和纬度数据。
        """
        with open(hdr_path, 'r') as f:
            lines = f.readlines()

        metadata = {}
        for line in lines:
            if '=' in line:
                key, value = line.split('=', 1)
                metadata[key.strip()] = value.strip()

        # 提取必要信息
        samples = int(metadata['samples'])
        lines = int(metadata['lines'])
        map_info = metadata['map info'].strip('{}').split(',')
        left_top_lon = float(map_info[3])
        left_top_lat = float(map_info[4])
        lon_res = float(map_info[5])
        lat_res = float(map_info[6])

        # 创建经纬度矩阵
        lon_matrix = np.zeros((lines, samples))
        lat_matrix = np.zeros((lines, samples))

        # 填充经纬度矩阵
        for i in range(lines):
            for j in range(samples):
                lon_matrix[i, j] = left_top_lon + j * lon_res
                lat_matrix[i, j] = left_top_lat - i * lat_res

        # 展开为两列
        lon_column = lon_matrix.flatten()
        lat_column = lat_matrix.flatten()

        return lon_column, lat_column

    @blinker(message="數據已成功保存為 列數據: {self.base_name}.txt")
    def ascii_to_column(self):
        """
        将 ASCII 文件转换为列数据格式。
        """

        #base, ext = os.path.splitext(output_path)

        # 读取 ASCII 文件内容
        with open(self.ascll, 'r') as f:
            file_content = f.read()
        # 分割文件内容为行列表
        file_lines = file_content.strip().split("\n")

        if os.path.exists(self.ascll):  # 确保文件存在
            os.remove(self.ascll)  # 删除临时文件


        # 初始化波段数据列表
        band_data_list = []
        current_band = []

        for line in file_lines:
            line = line.strip()
            if line.startswith(";") or not line:
                # 如果遇到空行，表示一个波段结束
                if not line and current_band:
                    band_data_list.append(current_band)
                    current_band = []
                continue
            # 分割行数据，假设数据之间用制表符或空格分隔
            line_data = line.split()
            current_band.append(line_data)

        # 添加最后一个波段的数据
        if current_band:
            band_data_list.append(current_band)

        # 确认所有波段的数据尺寸相同
        num_bands = len(band_data_list)
        rows = len(band_data_list[0])
        cols = len(band_data_list[0][0])

        # 将每个波段的数据展平成一维列表
        flattened_data_list = []
        for band in band_data_list:
            flattened_band = []
            for row in band:
                flattened_band.extend(row)
            flattened_data_list.append(flattened_band)

        # 转置数据，使每一行对应一个像素，列为各波段的值
        total_pixels = len(flattened_data_list[0])
        combined_data = []
        for i in range(total_pixels):
            pixel_values = [flattened_band[i] for flattened_band in flattened_data_list]
            combined_data.append(pixel_values)

        # 将 combined_data 转换为 NumPy 数组，数据类型为 float
        combined_data = np.array(combined_data, dtype=float)

        # 生成经度和纬度列
        lon_column, lat_column = self.generate_lat_lon(f'{self.base_name}.hdr')

        # 确保 lon_column 和 lat_column 的形状为 (total_pixels, 1)
        lon_column = lon_column.reshape(-1, 1)
        lat_column = lat_column.reshape(-1, 1)

        # 水平堆叠所有数据
        results = np.hstack((lon_column, lat_column, combined_data))

        df = pd.DataFrame(results) 
        condition = (df.iloc[:, 2:11] != 0).all(axis=1) & (df.iloc[:, 11] == 0)
        df_filtered = df[condition]
        lon_lat_fmt = ['%.2f', '%.2f']
        band_fmt = ['%d'] * combined_data.shape[1]
        fmt_list = lon_lat_fmt + band_fmt

        # 保存数据到文件
        np.savetxt(f"{self.base_name}.txt", df_filtered.values, delimiter='\t', fmt=fmt_list)
        self.yamls([self.info, 'ascii_to_column'], True)

    @blinker(message="{self.base_name}文件已成功轉化為 ASCII 格式。")
    def save_to_ascii(self):
        """
        将重投影后的数据保存为 ASCII 文件。
        """
        # 从重投影后的数据集中读取数据
        projected_ds = gdal.Open(self.base_name)

        # 获取波段数量
        projected_total_bands = projected_ds.RasterCount

        # 获取投影后的数据维度
        projected_cols = projected_ds.RasterXSize
        projected_rows = projected_ds.RasterYSize

        # 准备 ASCII 文件的头部信息
        header_lines = [
            ';',
            f'; ENVI ASCII Output of file: {self.base_name} [{datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")}]',
            f'; File Dimensions: {projected_cols} samples x {projected_rows} lines x {projected_total_bands} bands',
            '; Line Format    : ...',
            ';'
        ]

        # 创建一个临时文件
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file:
            # 获取临时文件路径
            temp_file_path = temp_file.name

            # 写入头部信息
            for line in header_lines:
                temp_file.write(line + '\n')

            # 遍历每个波段，处理数据并写入临时文件
            for band_idx in range(1, projected_total_bands + 1):
                band = projected_ds.GetRasterBand(band_idx)
                data = band.ReadAsArray()

                # 将 NaN 值替换为 0
                data = np.nan_to_num(data, nan=0)

                # 处理每一行数据
                for row in data:
                    row_str = '\t'.join(['{:.0f}'.format(x) for x in row])
                    temp_file.write(row_str + '\n')

                # 在每个波段之间添加一个空行
                temp_file.write('\n')

            # 将临时文件路径存储到 self.ascll 变量中
            self.ascll = temp_file_path

        self.yamls([self.info, 'save_to_ascii'], True)

    @blinker(message="文件已成功保存為 ENVI 格式到: {self.base_name}。")
    def save_to_raster(self):
        """
        将波段数据保存为 ENVI 格式的多波段栅格文件, 使用指定的地理坐标系统。
        """

        # 提取经纬度数据
        longitude_ds = self.band_data.pop('lon')
        latitude_ds = self.band_data.pop('lat')
        longitude_data = longitude_ds.ReadAsArray()  # Shape: (406, 271)
        latitude_data = latitude_ds.ReadAsArray()    # Shape: (406, 271)

        # 读取波段数据，处理多波段数据集
        data_bands = {}
        total_bands = 0

        for label, dataset in self.band_data.items():
            if label == 'TIR':
                # 仅读取波段7到12，命名为 TIR_27 到 TIR_32
                band_numbers = self.config.TIR_bands.band_numbers
                band_names = self.config.TIR_bands.band_names
                for band_number, band_name in zip(band_numbers, band_names):
                    band = dataset.GetRasterBand(band_number)
                    data = band.ReadAsArray()
                    data_bands[band_name] = data
                    total_bands += 1
            else:
                # 读取单波段数据集
                band = dataset.GetRasterBand(1)
                data = band.ReadAsArray()
                data_bands[label] = data
                total_bands += 1

        # 获取数据维度
        sample_band = next(iter(data_bands.values()))
        rows, cols = sample_band.shape

        # 插值经纬度数组到数据数组的尺寸
        lon = longitude_data
        lat = latitude_data

        # 原始经纬度网格
        lon_grid_y = np.linspace(0, lon.shape[0] - 1, lon.shape[0])
        lon_grid_x = np.linspace(0, lon.shape[1] - 1, lon.shape[1])

        # 目标网格
        target_grid_y = np.linspace(0, lon.shape[0] - 1, rows)
        target_grid_x = np.linspace(0, lon.shape[1] - 1, cols)
        target_meshgrid = np.meshgrid(target_grid_x, target_grid_y)
        target_points = np.stack((target_meshgrid[1].ravel(), target_meshgrid[0].ravel()), axis=-1)

        # 创建插值器
        lon_interpolator = RegularGridInterpolator((lon_grid_y, lon_grid_x), lon)
        lat_interpolator = RegularGridInterpolator((lon_grid_y, lon_grid_x), lat)

        # 插值
        lon_resampled = lon_interpolator(target_points).reshape(rows, cols)
        lat_resampled = lat_interpolator(target_points).reshape(rows, cols)

        # 创建内存数据集
        mem_driver = gdal.GetDriverByName('MEM')
        mem_ds = mem_driver.Create('', cols, rows, total_bands, gdal.GDT_Float32)

        # 写入每个波段数据到内存数据集
        for idx, (label, data) in enumerate(data_bands.items(), start=1):
            band = mem_ds.GetRasterBand(idx)
            band.WriteArray(data)
            band.SetDescription(label)  # 设置波段描述为指定的名称
            band.SetNoDataValue(np.nan)

        # 将插值后的经纬度数据写入临时文件
        temp_lon_file = tempfile.mktemp(suffix='.tif')
        temp_lat_file = tempfile.mktemp(suffix='.tif')
        driver = gdal.GetDriverByName('GTiff')
        lon_ds = driver.Create(temp_lon_file, cols, rows, 1, gdal.GDT_Float32)
        lon_ds.GetRasterBand(1).WriteArray(lon_resampled)
        lon_ds = None
        lat_ds = driver.Create(temp_lat_file, cols, rows, 1, gdal.GDT_Float32)
        lat_ds.GetRasterBand(1).WriteArray(lat_resampled)
        lat_ds = None

        # 设置地理定位信息
        mem_ds.SetMetadata({
            'X_DATASET': temp_lon_file,
            'X_BAND': '1',
            'Y_DATASET': temp_lat_file,
            'Y_BAND': '1',
            'PIXEL_OFFSET': '0',
            'LINE_OFFSET': '0',
            'PIXEL_STEP': '1',
            'LINE_STEP': '1',
        }, 'GEOLOCATION')

        # 定义目标投影（例如 EPSG:4326，WGS84）
        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(4326)

        # 确保输出目录存在
        output_dir = os.path.dirname(self.base_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 获取 ENVI 转换后的像素大小和范围（需要您提供）
        # 以下是示例值，请根据实际情况修改
        x_min, x_max = lon_resampled.min(), lon_resampled.max()
        y_min, y_max = lat_resampled.min(), lat_resampled.max()

        # 计算像素大小
        x_res = (x_max - x_min) / (cols - 1)
        y_res = (y_max - y_min) / (rows - 1)


        # 设置输出范围和分辨率
        warp_options = gdal.WarpOptions(
            format='ENVI',
            dstSRS=target_srs.ExportToWkt(),
            srcNodata=np.nan,
            dstNodata=np.nan,
            geoloc=True,
            outputBounds=(x_min, y_min, x_max, y_max),
            xRes=x_res,
            yRes=y_res,
            resampleAlg='near',
            multithread=True,
            options=['INTERLEAVE=BSQ']
        )

        # 使用 gdal.Warp 进行重投影并保存为 ENVI 格式
        gdal.Warp(
            self.base_name,
            mem_ds,
            options=warp_options
        )

        # 删除临时文件
        os.remove(temp_lon_file)
        os.remove(temp_lat_file)

        self.yamls([self.info, 'save_to_raster'], True)

    @blinker(message="處理{self.base_name}文件後，新增了{new_count}條數據, 合并了{merge_count}, 現在共有{new_total_count}條數據。")
    def append_data(self, target_file, haxi=None):
        if haxi is None:
            """
            将新数据追加到目标文件末尾，不处理重复坐标。
            """
            original_count = 0
            new_count = 0
            merge_count = '无'

            # 1. 读取原始文件数据量
            if os.path.exists(target_file):
                with open(target_file, 'r') as f:
                    original_count = len(f.readlines())
                #print(f"原始文件 {target_file} 包含 {original_count} 行数据。")
            #else:
                #open(target_file, 'w').close()  # 创建空文件
                #print(f"目标文件 {target_file} 创建成功，文件是新的。")

            # 2. 读取并追加新数据
            new_file = f"{self.base_name}.txt"
            if not os.path.exists(new_file):
                #print(f"新数据文件 {new_file} 不存在，跳过追加。")
                return

            with open(new_file, 'r') as f:
                new_lines = f.readlines()
                new_count = len(new_lines)
            
            # 追加写入新数据
            with open(target_file, 'a') as target:
                target.writelines(new_lines)
            
            # 3. 验证结果
            new_total_count = original_count + new_count
            #print(f"写入完成，新增了 {new_count} 个数据点。原始数据量为 {original_count} 行，现在为 {new_total_count} 行。")
            #print(f'新增数据加上原始数据是否等于现在数据：{new_count + original_count == new_total_count}')
            
            # 强制验证（理论上应该永远成立）
            if new_count + original_count != new_total_count:
                #print(f"数据多出了 {new_total_count - (new_count + original_count)} 行，请检查文件写入。")
                raise ValueError("数据完整性验证失败")
  
        
        
        else:

            """
            将新数据合并到目标文件中，确保正确读取原数据并合并。
            """
            # 初始化哈希表并读取目标文件数据
            self.geo_index = {}
            original_geo_count = 0

            # 检查目标文件是否存在并读取数据
            if os.path.exists(target_file):
                with open(target_file, 'r') as target:
                    target_lines = target.readlines()
                original_count = len(target_lines)
                print(f"原始文件 {target_file} 包含 {original_count} 行数据。")

                # 构建哈希表
                for line in target_lines:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        print(f"忽略无效行：{line.strip()}")
                        continue
                    try:
                        lon, lat = map(float, parts[:2])
                        values = list(map(float, parts[2:]))
                        self.geo_index[(lon, lat)] = values
                    except ValueError as e:
                        print(f"解析行 '{line.strip()}' 失败：{e}")
                        continue
                original_geo_count = len(self.geo_index)
                print(f"哈希表构建完成，包含 {original_geo_count} 个唯一数据点。")
            else:
                original_count = 0
                with open(target_file, 'w') as target:
                    pass  # 创建空文件
                print(f"目标文件 {target_file} 不存在，已创建新文件。")

            # 处理新数据
            new_file = f"{self.base_name}.txt"
            if not os.path.exists(new_file):
                print(f"新数据文件 {new_file} 不存在，跳过合并。")
                return

            with open(new_file, 'r') as new_data:
                new_lines = new_data.readlines()
                print(f"读取新数据文件 {new_file}，共 {len(new_lines)} 行。")

            merge_count = 0
            new_count = 0

            for line in new_lines:
                parts = line.strip().split()
                if len(parts) < 2:
                    print(f"忽略无效新数据行：{line.strip()}")
                    continue
                try:
                    new_lon, new_lat = map(float, parts[:2])
                    new_values = list(map(float, parts[2:]))
                except ValueError as e:
                    print(f"解析新数据行 '{line.strip()}' 失败：{e}")
                    continue

                key = (new_lon, new_lat)
                if key in self.geo_index:
                    # 合并数据：计算平均值
                    existing = self.geo_index[key]
                    merged = [(e + n) / 2 for e, n in zip(existing, new_values)]
                    self.geo_index[key] = merged
                    merge_count += 1
                else:
                    # 新增数据
                    self.geo_index[key] = new_values
                    new_count += 1

            # 写入合并后的数据
            with open(target_file, 'w') as target:
                for (lon, lat), values in self.geo_index.items():
                    line = f"{lon}\t{lat}\t" + "\t".join(map(str, values)) + "\n"
                    target.write(line)
            print(f"数据合并完成。合并 {merge_count} 处，新增 {new_count} 处。")

            # 验证数据完整性
            new_total_count = len(self.geo_index)
            expected_count = original_geo_count + new_count
            if new_total_count != expected_count:
                print(f"数据不一致！预期 {expected_count}，实际 {new_total_count}，差异 {new_total_count - expected_count}")
                raise ValueError("数据合并验证失败。")
            else:
                print("数据验证通过。")

        # 其他后续操作（如有）
        self.yamls([self.info, 'append_data'], True)
        return {'new_count': new_count, 'merge_count': merge_count, 'new_total_count': new_total_count}  # 装饰器返回值
  
    @blinker(message="處理文件後，白天文件已保存至{save_day_path}。夜晚文件已保存至{save_night_path}。")
    def radiometric_to_bt(self, file_path):
        # 从YAML文件加载参数
        # h = 6.62607015e-34  # 普朗克常�?, 单位: J·s
        # c = 2.99792458e8    # 光�?, 单位: m/s
        # k = 1.380649e-23    # 玻尔兹曼常数, 单位: J/K
        # modis_wavelengths = np.array([6.715, 7.325, 8.550, 9.730, 11.030, 12.020]) * 1e-6 # 波段中心波长，单位转换为�?
        # modis_scales = np.array([0.000117557, 0.00019245, 0.000532487, 0.000406323, 0.000840022, 0.000729698])   #27, 28,19,30,31,32
        # modis_offsets = np.array([2730.5833, 2317.4883, 2730.5835, 1560.3333, 1577.3397, 1658.2213])  #27, 28,19,30,31,32
        
        def 转换(file_path):   #核心转换
            df = pd.read_csv(file_path, sep="\t", header=None)

            h = float(self.config.constants.planck_constant)
            c = float(self.config.constants.speed_of_light)
            k = float(self.config.constants.boltzmann_constant)

            # 加载MODIS参数
            modis_params = self.config.modis_parameters
            modis_wavelengths = np.array(modis_params.wavelengths)
            modis_scales = np.array(modis_params.radiance_scales)
            modis_offsets = np.array(modis_params.radiance_offsets)

            emis_slope = modis_params.emissivity_coefficients.slope
            emis_intercept = modis_params.emissivity_coefficients.intercept
            lst_slope = modis_params.lst_coefficients.slope

            # 辐射定标计算
            selected_bands = df.iloc[:, 2:8]  # 选择第3到第8列（Python是0-based索引）
            modis_radiance = modis_scales * (selected_bands - modis_offsets)

            # 亮温转换计算
            wavelength_expanded = modis_wavelengths.reshape(1, -1)
            numerator = h * c
            denominator = k * wavelength_expanded
            log_term = np.log((2 * h * c**2) / (wavelength_expanded**5 * modis_radiance * 1e6) + 1)
            modis_em_temp = np.where(modis_radiance <= 0, 0, numerator / (denominator * log_term))

            # 数据格式化
            ll = df.iloc[:, :2]  # 经纬度列
            modis_em_temp = modis_em_temp.round(3)
            emis = (emis_slope * df.iloc[:, 7:9] + emis_intercept).round(3)
            lst = (lst_slope * df.iloc[:, 10]).round(2)
            qc = df.iloc[:, 11].round(0)

            # 合并数据
            data = pd.concat([pd.DataFrame(ll), pd.DataFrame(modis_em_temp), pd.DataFrame(emis), pd.DataFrame(lst), pd.DataFrame(qc)], axis=1)
            # 生成输出文件路径
            save_path = os.path.join(os.path.split(file_path)[0], f"bt_{os.path.split(file_path)[1]}")

            # 保存结果
            data.to_csv(save_path, sep='\t', header=False, index=False)
            return save_path # 装饰器返回值

        base_day_path = f'proc_{file_path}/L_daycol.txt'
        base_night_path = f'proc_{file_path}/L_nightcol.txt'
        save_day_path = 转换(base_day_path)
        save_night_path = 转换(base_night_path)

        return {'save_day_path': save_day_path, 'save_night_path': save_night_path}  # 装饰器返回值



    @blinker(message="合并白天{count_day}天的数据到{output_day_file}文件,大小为。{shape_day}。\n\t\t\t\t  合并夜晚{count_night}天的数据到{output_night_file}文件,大小为。{shape_night}。")
    def merge_files(self, base_path):
        base_path = f'proc_{base_path}'
        merged_day_data = []
        merged_night_data = []
        count_day = 0
        count_night = 0
        for subdir in os.listdir(base_path):
            subdir_path = os.path.join(base_path, subdir)
            daycol_file_path = os.path.join(subdir_path, 'daycol.txt')  # 优化此行
            if os.path.isfile(daycol_file_path):  # 检查文件是否存在
                with open(daycol_file_path, 'r') as file:
                    merged_day_data.extend(file.readlines())
                count_day += 1
            nightcol_file_path = os.path.join(subdir_path, 'nightcol.txt')  # 优化此行
            if os.path.isfile(nightcol_file_path):  # 检查文件是否存在
                with open(nightcol_file_path, 'r') as file:
                    merged_night_data.extend(file.readlines())
                count_night += 1
        output_day_file = os.path.join(base_path, 'L_daycol.txt')
        output_night_file = os.path.join(base_path, 'L_nightcol.txt')
        with open(output_day_file, 'w') as file:
            file.writelines(merged_day_data)
        with open(output_night_file, 'w') as file:
            file.writelines(merged_night_data)

        shape_day = pd.DataFrame(merged_day_data).shape
        shape_night = pd.DataFrame(merged_night_data).shape

        return {'count_day': count_day, 'output_day_file': output_day_file, 'shape_day': shape_day,
                'count_night': count_night, 'output_night_file': output_night_file, 'shape_night':shape_night}  # 装饰器返回值

    def core(self, base_path, date):
        # 查找匹配的文件
        self.find_matching_files(base_path, date)
        for matches in self.Matches:
            print(f'\n當前處理文件類型是：{matches[0]} 天\n')
            self.yamls = EasyDict_gu(matches[3])

            for match in matches[1]:  
                self.base_name = f'{match[3]}/{match[0]}'
                self.info = matches[0], f't{match[0]}'
                if self.yamls([self.info, 'append_data']) is True:
                    print(f"文件 {match[0]} 已經處理過，跳過此文件。")
                    continue
                print("\n當前處理文件是 Time:", match[0], "\n TIR:", match[1], "\n LST:", match[2], "\n Save Folder:", match[3])
                self.load_modis_data(match)  # 加载MODIS数据
                self.save_to_raster()        # 保存为ENVI格式
                self.save_to_ascii()         # 保存为ASCII格式
                self.ascii_to_column()       # 转换为列数据格式
                self.append_data(matches[2]) # 追加数据到文件
            print(f'\n{matches[0]} 天的所有文件已经处理完畢。')

    @计时器
    def batch_core(self, base_path):
        for dir_name in os.listdir(base_path):
                if os.path.isdir(os.path.join(base_path, dir_name)):
                    print(f"\n当前处理的文件日期是: {dir_name}")  # 获取并打印文件夹名字
                    self.core(base_path, dir_name)
        print(f'\n{base_path}文件夹下的所有文件已经处理完畢。\n')
        print(f'\t猫迪司（modis）最終合併輻射定標段落')
        self.merge_files(base_path)   # 分别合并多日白天与晚上所有文件
        self.radiometric_to_bt(base_path)  # 将对应的白天或晚上的文件转换为亮温数据
        print(f'\t猫迪司（modis）处理完毕。')

if __name__ == '__main __': 
    #猫迪大祭司().core('data', '230806')
    猫迪大祭司().batch_core('datatest')

#输入数据的总文件夹，文件夹下有多个日期的文件夹，每个日期的文件夹下有白天和晚上文件夹，白天或晚上文件夹下有LST和TIR文件夹，LST和TIR文件夹下有多个文件
#输出结果是将总文件夹下所有白天和晚上文件夹下的LST和TIR文件合并到一个文件中，然后将合并后的文件转换为BT文件
#例如输入总文件夹为data，data下有230806，230807，230808三个文件夹，230806下有day和night两个文件夹，day和night下有LST和TIR文件夹，LST和TIR文件夹下有多个文件
#输出结果是将在proc_data文件夹下proc_datatest/bt_L_daycol.txt和proc_datatest/bt_L_nightcol.txt文件中

