import seaborn as sns
import dill
import pathlib

if __name__ == '__main__':
    res_path = pathlib.Path('./case0_res.pkl').absolute()
    with open(res_path, 'rb') as f:
        res = dill.load(f)
        
    xs = res['xs']
    modes = res['modes']
    