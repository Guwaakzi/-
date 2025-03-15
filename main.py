import time

from deal import 猫迪大祭司

猫迪大祭司().batch_core('datatest')

#參數配置文件在config/config.yaml中

#输入数据的总文件夹，文件夹下有多个日期的文件夹，每个日期的文件夹下有白天和晚上文件夹，白天或晚上文件夹下有LST和TIR文件夹，LST和TIR文件夹下有多个文件
#输出结果是将总文件夹下所有白天和晚上文件夹下的LST和TIR文件合并到一个文件中，然后将合并后的文件转换为BT文件
#例如输入总文件夹为data，data下有230806，230807，230808三个文件夹，230806下有day和night两个文件夹，day和night下有LST和TIR文件夹，LST和TIR文件夹下有多个文件
#输出结果是将在proc_data文件夹下proc_datatest/bt_L_daycol.txt和proc_datatest/bt_L_nightcol.txt文件中