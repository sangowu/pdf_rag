from paddleocr import PPStructureV3
# 例：中文文档，使用默认 PP-StructureV3 产线
pipeline = PPStructureV3(lang="ch")
# 导出当前产线完整配置到 yaml 文件
pipeline.export_paddlex_config_to_yaml("config/PP-StructureV3.yaml")