import csv

# 输入和输出文件路径
input_file = 'a.csv'
output_file = 'filtered_bwd_config.csv'

# 需要提取的列
fields_to_extract = ['GrdX', 'GrdY', 'GrdZ', 'BlkX', 'BlkY', 'BlkZ', 'DymSMem (MB)']

with open(input_file, 'r') as csv_in, open(output_file, 'w', newline='') as csv_out:
    reader = csv.DictReader(csv_in)
    writer = csv.DictWriter(csv_out, fieldnames=fields_to_extract)
    
    # 写入表头
    writer.writeheader()
    
    # 提取第一条包含 'bwd' 的数据
    for row in reader:
        if 'bw' in row['Name']:
            filtered_row = {field: row[field] for field in fields_to_extract}
            writer.writerow(filtered_row)
            break  # 找到第一条后退出循环

print(f"提取完成，结果已保存至 {output_file}")
