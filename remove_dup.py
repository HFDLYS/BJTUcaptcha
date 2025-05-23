import pandas as pd

def remove_duplicates_by_column(input_csv, output_csv, subset_columns):
    """
    根据指定列去除重复行
    :param input_csv: 输入文件路径
    :param output_csv: 输出文件路径
    :param subset_columns: 需要检查重复的列名列表，如 ['id', 'name']
    """
    df = pd.read_csv(input_csv)

    # 检查列名是否存在
    missing_cols = [col for col in subset_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV文件中缺少以下列: {missing_cols}")

    # 按指定列去重
    df.drop_duplicates(subset=subset_columns, keep='first', inplace=True)

    df.to_csv(output_csv, index=False)
    print(f"基于列 {subset_columns} 去重完成！原始行数: {len(df) + len(df.duplicated(subset=subset_columns))}, 去重后行数: {len(df)}")

remove_duplicates_by_column("test.csv", "output_no_duplicates.csv", ['image_name'])

