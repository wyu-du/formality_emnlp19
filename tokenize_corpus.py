from utils.multi_process_tokenizer import tokenizer
from utils.file_api import read_file_lines,write_file_lines
domains = ['flickr']


def tok():
    ori_path = 'training_data/'
    in_files = []
    out_files = []
    for domain in domains:
        in_files.append(ori_path + 'ori/flickr' + '/flickr-funny-train.txt')
        in_files.append(ori_path + 'ori/flickr' + '/flickr-funny-test.txt')
        in_files.append(ori_path + 'ori/flickr' + '/flickr-funny-val.txt')
        in_files.append(ori_path + 'ori/flickr' + '/flickr-romantic-train.txt')
        in_files.append(ori_path + 'ori/flickr' + '/flickr-romantic-test.txt')
        in_files.append(ori_path + 'ori/flickr' + '/flickr-romantic-val.txt')
        
#        in_files.append(ori_path + 'dif_models_' + domain + '/sentiment.test.0')
#        in_files.append(ori_path + 'dif_models_' + domain + '/sentiment.dev.0')
#        in_files.append(ori_path + 'dif_models_' + domain + '/sentiment.train.1')
#        in_files.append(ori_path + 'dif_models_' + domain + '/sentiment.test.1')
#        in_files.append(ori_path + 'dif_models_' + domain + '/sentiment.dev.1')
        out_files.append(ori_path + 'dif_models_' + domain + '/funny.train.tok')
        out_files.append(ori_path + 'dif_models_' + domain + '/funny.test.tok')
        out_files.append(ori_path + 'dif_models_' + domain + '/funny.val.tok')
        out_files.append(ori_path + 'dif_models_' + domain + '/romantic.train.tok')
        out_files.append(ori_path + 'dif_models_' + domain + '/romantic.test.tok')
        out_files.append(ori_path + 'dif_models_' + domain + '/romantic.val.tok')
    for f_in,f_out in zip(in_files,out_files):
        sens=read_file_lines(f_in)
        s_tok=tokenizer(sens,type='word',join=True)
        write_file_lines(path=f_out,lines=s_tok)

if __name__=='__main__':
    tok()
    print("all work has finished")



