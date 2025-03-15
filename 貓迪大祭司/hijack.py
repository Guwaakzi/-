import sys
import time
import threading
from functools import wraps
import yaml
import random
from easydict import EasyDict
from collections import defaultdict
from datetime import datetime

def ryaml(file_path):
    with open(file_path, "r", encoding='utf-8') as file:
        config_data = yaml.safe_load(file)
    return EasyDict(config_data)

def dyaml(edict):
    if isinstance(edict, EasyDict):
        edict = {k: dyaml(v) for k, v in edict.items()}
    return edict


def 计时器(func):
    """
    计算函数运行时间的装饰器
    输出格式：
    开始时间: YYYY-MM-DD HH:MM:SS
    结束时间: YYYY-MM-DD HH:MM:SS
    共用时: X小时Y分钟Z秒
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 记录开始时间
        开始时间 = time.time()
        开始时间格式化 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"开始时间: {开始时间格式化}")

        # 执行函数
        result = func(*args, **kwargs)

        # 记录结束时间
        结束时间 = time.time()
        结束时间格式化 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"开始时间: {开始时间格式化}")
        print(f"结束时间: {结束时间格式化}")

        # 计算总运行时间
        总时长 = 结束时间 - 开始时间
        小时 = int(总时长 // 3600)
        分钟 = int((总时长 % 3600) // 60)
        秒 = int(总时长 % 60)
        print(f"共用时: {小时}小时{分钟}分钟{秒}秒")

        return result
    return wrapper

def blinker(message=None):
    colors = [
            "\033[38;5;196m",  # 红色
            "\033[38;5;202m",  # 橙红色
            "\033[38;5;208m",  # 橙色
            "\033[38;5;214m",  # 浅橙色
            "\033[38;5;220m",  # 金黄色
            "\033[38;5;226m",  # 黄色
            "\033[38;5;190m",  # 黄绿色
            "\033[38;5;154m",  # 浅绿色
            "\033[38;5;118m",  # 绿色
            "\033[38;5;82m",   # 草绿色
            "\033[38;5;46m",   # 翠绿色
            "\033[38;5;47m",   # 淡绿色
            "\033[38;5;48m",   # 绿松石色
            "\033[38;5;49m",   # 浅青色
            "\033[38;5;51m",   # 天蓝色
            "\033[38;5;87m",   # 粉蓝色
            "\033[38;5;123m",  # 蓝绿色
            "\033[38;5;159m",  # 蓝色
            "\033[38;5;195m",  # 淡蓝色
            "\033[38;5;231m",  # 白色
            "\033[38;5;225m",  # 浅紫色
            "\033[38;5;219m",  # 粉紫色
            "\033[38;5;213m",  # 浅粉红色
            "\033[38;5;207m",  # 洋红色
            "\033[38;5;201m",  # 玫红色
            "\033[38;5;165m",  # 紫红色
            "\033[38;5;129m",  # 紫色
            "\033[38;5;93m",   # 深紫色
            "\033[38;5;57m",   # 蓝紫色
            "\033[38;5;21m",   # 深蓝色
            "\033[38;5;27m",   # 靛蓝色
            "\033[38;5;33m",   # 深青色
            "\033[38;5;39m",   # 浅青蓝色
            "\033[38;5;45m",   # 浅绿色
            "\033[38;5;51m",   # 天蓝色
            "\033[0m"          # 重置颜色
        ]
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            # 启动闪烁
            stop_event = threading.Event()

            def _blink():
                

                i = 0
                while not stop_event.is_set():
                    symbol = "█" * (i % 64 + 1)  # 生成 1 到 5 个方块
                    color = colors[i % len(colors)]
                    sys.stdout.write(f'\r{color}{func.__name__} {symbol}\033[0m')
                    sys.stdout.flush()
                    time.sleep(0.01)
                    i += 1

            blinking_thread = threading.Thread(target=_blink)
            blinking_thread.start()

            try:
                # 执行原始函数
                result = func(*args, **kwargs) or {}
                dynamic_message = message
                if message and args and hasattr(args[0], "__dict__"):
                    for attr, value in vars(args[0]).items():
                        placeholder = f"{{self.{attr}}}"
                        if placeholder in dynamic_message:
                            dynamic_message = dynamic_message.replace(placeholder, str(value))
            finally:
                # 停止闪烁
                stop_event.set()
                blinking_thread.join()
                # 清除闪烁字符并打印完成信息
                sys.stdout.write('\r' + ' ' * 80 + '\r')
                sys.stdout.flush()
                time.sleep(0.05) 

                color = random.choice(colors)
                elapsed_time = time.time() - start_time
                hours, remainder = divmod(elapsed_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                formatted_message = f"{dynamic_message.format(**{**globals(), **result})}" if dynamic_message else ""
                print(f"{color}\t{func.__name__}函數 執行完成。{formatted_message} 耗時總計 {int(hours)} 小時 {int(minutes)} 分 {seconds:.2f} 秒。")  #\033[0m 重置颜色

            return result
        return wrapper
    return decorator

sti = 234124
# 使用装饰器
@blinker(message="test测试信息显示成功，其中的变量名time是 {time2}, tim是 {tim}, sti是 {sti}")
#@blinker()
def test():
    time2  = 1
    tim = 23
    # 模拟耗时任务
    time.sleep(time2)
    return {"time2": time2, "tim": tim}
    
    #

if __name__ == "__main__":
    test()


class RecursiveDefaultDict(defaultdict):
    def __init__(self, parent=None):
        super().__init__(lambda: RecursiveDefaultDict(self.top_parent))
        self.parent = parent
        self.top_parent = parent if parent is None else parent.top_parent

    def __getattr__(self, item):
        if item not in self:
            self[item] = RecursiveDefaultDict(self.top_parent)
        return self[item]

    def __setattr__(self, key, value):
        if key in {'parent', 'top_parent'}:
            super().__setattr__(key, value)
        else:
            self[key] = value
            if self.top_parent:
                self.top_parent.save()

class EasyDict_gu:
    def __init__(self, file_path):
        self.file_path = file_path
        self.top_parent = self
        self.data = RecursiveDefaultDict(self)
        self.load()

    def load(self):
        """Load data from the YAML file."""
        try:
            with open(self.file_path, 'r') as file:
                loaded_data = yaml.safe_load(file)
                if loaded_data:
                    self._populate(self.data, loaded_data)
        except FileNotFoundError:
            print(f"No file found. Starting with an empty dictionary.")

    def _populate(self, dict_obj, data):
        """Recursively populate the nested structure from loaded data."""
        for key, value in data.items():
            if isinstance(value, dict):
                self._populate(dict_obj[key], value)
            else:
                dict_obj[key] = value

    def to_plain_dict(self, obj):
        """Convert RecursiveDefaultDict to a plain dictionary."""
        if isinstance(obj, defaultdict):
            plain_dict = {}
            for k, v in obj.items():
                plain_dict[k] = self.to_plain_dict(v)
            return plain_dict
        else:
            return obj

    def save(self):
        """Save the current structure to the YAML file."""
        if self.file_path:
            with open(self.file_path, 'w') as file:
                plain_data = self.to_plain_dict(self.data)
                yaml.dump(plain_data, file, default_flow_style=False, allow_unicode=True)

    def __call__(self, keys, value=None):
        """Set or get a value in the nested structure using a list of keys."""
        expanded_keys = []
        for key in keys:
            if isinstance(key, tuple):  # 自动展开元组
                expanded_keys.extend(key)
            else:
                expanded_keys.append(key)

        current = self.data
        for key in expanded_keys[:-1]:
            if key not in current:
                if value is None:  # 如果是获取操作，路径不存在则返回 None
                    return None
                current[key] = RecursiveDefaultDict(self.top_parent)
            current = current[key]

        if value is not None:  # 设置值
            current[expanded_keys[-1]] = value
            self.save()
            return value
        else:  # 获取值
            return current.get(expanded_keys[-1], None)


    def get_nested(self, keys):
        """Get a value from the nested structure using a list of keys."""
        current = self.data
        for key in keys:
            current = current[key]
        return current

    def __getattr__(self, item):
        if item not in self.__dict__:
            return self.data.__getattr__(item)
        return self.__dict__[item]

    def __setattr__(self, key, value):
        if key in ['data', 'file_path', 'top_parent']:
            super().__setattr__(key, value)
        else:
            self.data.__setattr__(key, value)

# 使用示例
if __name__ == "__main__":
    file_path = "config.yaml"
    config = EasyDict_gu(file_path)

    # 情况1：通过动态路径设置设备采集数据（包含元组展开）
    year1, timestamp1 = '2025', '1205'
    device_info = year1, timestamp1  # 表示设备采集的年份和具体时间
    config([device_info, 'sensor_1', 'temperature'], True)  # 存储温度传感器数据

    # 情况2：通过动态路径直接设置设备采集数据
    year, timestamp = 'hunian', 'zhengyue'
    config([year, timestamp, 'jieri', 'chunjie'], True)  # 存储湿度传感器数据

    # 情况3：通过直接属性访问设置设备的其他元信息
    config.device_2.sensor_2.location = "Room A"  # 记录传感器的位置
    config.device_2.sensor_2.status = True  # 记录传感器状态

    # 情况4：判断动态路径中的值是否符合条件（包含元组展开）
    if config([device_info, 'sensor_1', 'temperature']) is True:
        print("判斷成功")
    else:
        print(f'实际值是{config([device_info, "sensor_1", "temperature"])}')

    # 情况5：判断动态路径中的值是否符合条件
    if config([year, timestamp, 'jieri', 'chunjie']) is True:
        print("判斷成功")
    else:
        print(f'实际值是{config([year, timestamp, "sensor_2", "humidity"])}')

    #情况6：判断直接属性访问中的值是否符合条件
    if config.device_2.sensor_2.status is True:
        print("判斷成功")
    else:
        print(f'实际值是{config.device_2.sensor_2.status}')