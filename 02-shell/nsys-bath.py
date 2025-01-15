import os
import csv
import argparse
import subprocess

def parse_launch_script(script_path):
    """
    从启动脚本中解析参数
    """
    params_list = []
    with open(script_path, 'r') as file:
        for line in file:
            # 去掉换行符和多余的空格
            line = line.strip()
            if line.startswith('./cudnn.bin'):
                # 提取参数部分
                args = line.split()[1:]  # 去掉 './cudnn.bin'
                params_list.append(args)
    return params_list

def run_nsys_profile(application, params, output_report):
    """
    运行 nsys profile 命令生成报告文件
    """
    command = ['nsys', 'profile', '--output', output_report, application] + params
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, check=True)

def run_nsys_stats(report_file, output_csv):
    """
    运行 nsys stats 命令生成 CSV 文件
    """
    command = ['nsys', 'stats', '--report', 'cuda_gpu_trace', '--format', 'csv', '--output', output_csv, report_file]
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, check=True)

def process_csv_files(input_csv, param, merged_writer):
    """
    处理生成的 CSV 文件并提取所需信息，将结果写入合并的 CSV 文件
    """
    # 打开并读取 CSV 文件
    with open(input_csv, 'r') as infile:
        reader = csv.DictReader(infile)
        
        # 遍历每一行数据
        for row in reader:
            # 检查 Name 列是否包含 'bw'
            if 'bw' in row.get('Name', ''):
                # 提取所需的列
                grd_x = row.get('GrdX', 'N/A')
                grd_y = row.get('GrdY', 'N/A')
                grd_z = row.get('GrdZ', 'N/A')
                blk_x = row.get('BlkX', 'N/A')
                blk_y = row.get('BlkY', 'N/A')
                blk_z = row.get('BlkZ', 'N/A')
                
                # 写入合并的 CSV 文件
                merged_writer.writerow({
                    '参数': ' '.join(param),  # 将参数列表转换为字符串
                    'GrdX': grd_x,
                    'GrdY': grd_y,
                    'GrdZ': grd_z,
                    'BlkX': blk_x,
                    'BlkY': blk_y,
                    'BlkZ': blk_z
                })
                break  # 只取一行，然后退出循环

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Automate nsys profiling and process CSV files.')
    parser.add_argument('application', type=str, help='Path to the CUDA application (e.g., ./cudnn.bin)')
    parser.add_argument('launch_script', type=str, help='Path to the launch script containing parameters')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to store output files')
    parser.add_argument('--merged_output', type=str, default='merged_output.csv', help='Path to the merged output CSV file')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 打开合并的 CSV 文件并写入列名
    with open(args.merged_output, 'w', newline='') as merged_file:
        merged_columns = ['参数', 'GrdX', 'GrdY', 'GrdZ', 'BlkX', 'BlkY', 'BlkZ']
        merged_writer = csv.DictWriter(merged_file, fieldnames=merged_columns)
        merged_writer.writeheader()
        
        # 从启动脚本中解析参数
        params_list = parse_launch_script(args.launch_script)
        
        # 遍历每个参数
        for params in params_list:
            # 生成唯一的标识符（例如参数的哈希值）
            param_id = '_'.join(params)  # 将参数列表转换为字符串作为唯一标识
            report_file = os.path.join(args.output_dir, f'report_{param_id}.nsys-rep')
            output_csv = os.path.join(args.output_dir, f'output_{param_id}')
            final_output_csv = output_csv +'_cuda_gpu_trace.csv'
            
            # 运行 nsys profile
            run_nsys_profile(args.application, params, report_file)
            
            # 运行 nsys stats
            run_nsys_stats(report_file, output_csv)
            
            # 处理生成的 CSV 文件并将结果写入合并的 CSV 文件
            process_csv_files(final_output_csv, params, merged_writer)
            
            print(f"Processed parameters: {' '.join(params)}")
    
    print(f"Merged output saved to {args.merged_output}")

if __name__ == '__main__':
    main()