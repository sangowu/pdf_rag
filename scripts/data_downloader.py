import openxlab
openxlab.login(ak="kwvpj9orq8lmqz4obmla", sk="z7x3r5eyg0lm4bjdd8ylzoympdjokp6yamnzqevr") # 进行登录，输入对应的AK/SK，可在个人中心添加AK/SK

from openxlab.dataset import get
get(dataset_repo='OpenDataLab/OmniDocBench', target_path='./data/raw/OpenDataLab___OmniDocBench') # 数据集下载
