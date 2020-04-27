import os,shutil
from gpt.src.multi_gpu_training import multi_gpu_trainer
from gpt.src.gpt2 import GPT2
from gpt.src.single_gpu_serving import beam_search_generator,ensemble_beam_search_generator
from utils.file_api import read_file_lines,write_file_lines
from utils.cat_files import cat_files
from gpt.src import encoder
from gpt.config import *



def train(train_corpus,dev_corpus,infer_ckpt_path,train_ckpt_path,sep_flag='\t',sep_num=1,
          learning_rate=1e-5, init_step_num=1, batch_size=128, mini_batch=16,
          eval_per_n_steps=10,total_steps=30000, early_stop_steps=50,
          max_to_save=1, append_eos=True,
          eos_symbol='\n'):
    gpt2 = GPT2(config_path)
    trainer = multi_gpu_trainer(device_id=[0], model_fn=gpt2)
    trainer.build_data_parallel_training_graph()
    trainer.only_predict_target=True
    trainer.sep_flag=sep_flag
    trainer.sep_num=sep_num
    trainer.training(train_corpus=train_corpus,
                     dev_corpus=dev_corpus,
                     infer_ckpt_path=infer_ckpt_path, train_ckpt_path=train_ckpt_path,
                     learning_rate=learning_rate, init_step_num=init_step_num,
                     batch_size=batch_size, mini_batch=mini_batch,
                     eval_per_n_steps=eval_per_n_steps,
                     total_steps=total_steps,
                     early_stop_steps=early_stop_steps,
                     max_to_save=max_to_save,
                     append_eos=append_eos,
                     eos_id=gpt2.text_enc.encode(eos_symbol)[0])


def test(model_dir,input_path,output_path,beam_size=4,max_dec_len=60,dec_alpha=0.6):
    gpt2 = GPT2(config_path)
    generator = beam_search_generator(gpt2, beam_size=beam_size,
                                      model_directory=model_dir, max_dec_len=max_dec_len,
                                      dec_alpha=dec_alpha)
    sess=generator.build_graph_and_restore(eos_id=gpt2.text_enc.encode('\n')[0])
    lines=read_file_lines(input_path)
    result=[]
    for line in lines:
        text = generator.generate(sess,line)
        if len(text) == 0: text = 'null'
        result.append(text)
    sess.close()
    write_file_lines(output_path, result)


def ensemble_test(domain='fr',model_type=['ori','rule'],
                  beam_size=4,max_dec_len=60,dec_alpha=0.6):
    model_dir=['./models_'+domain+'/'+t+'/formality_infer/' for t in model_type]
    input_path=['../training_data/dif_models_'+domain+'/eval.'+t for t in model_type]
    output_path = '../evaluate/gyafc_model_outputs/'+domain+'_out/formal.gpt.'+'_'.join(model_type)+'.ens'
    gpt2 = GPT2(config_path)
    generator = ensemble_beam_search_generator(gpt2, beam_size=beam_size,
                                      model_directorys=model_dir, max_dec_len=max_dec_len,
                                      dec_alpha=dec_alpha)
    sess=generator.build_graph_and_restore(eos_id=gpt2.text_enc.encode('\n')[0],model_num=len(model_type))
    lines=[read_file_lines(p) for p in input_path]
    result=[]
    line_len=[len(l) for l in lines]
    max_l,min_l=max(line_len),min(line_len)
    assert max_l==min_l
    for i in range(0,max_l):
        result.append(generator.generate(sess,[lines[j][i] for j in range(0,len(model_type))]))
    sess.close()
    write_file_lines(output_path, result)


def simple_finetune(domain='fr',methods='ori',max_len_limit=220):
    methods=[methods]
    if not os.path.exists('gpt/models/'+domain):
        os.mkdir('gpt/models/'+domain)
    model_path='gpt/models/'+domain+'/'+'_'.join(methods)
    output_path='evaluate/'+domain+'/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
        os.mkdir(model_path+'/modern_train')
        os.mkdir(model_path+'/modern_infer')
    data_path = 'training_data/dif_models_'+domain+'/'
    cat_files([data_path + 'original-train.tok']+ [data_path + 'modern-train.tok'],
              data_path + 'train.'+'_'.join(methods),
              tokenizer=text_enc, max_len=max_len_limit)
    cat_files([data_path + 'original-val.tok'] + [data_path + 'modern-val.tok'],
              data_path + 'val.' + '_'.join(methods),
              tokenizer=text_enc, max_len=max_len_limit)
    lp = cat_files([data_path + 'original-test.tok'],
                   data_path + 'eval.' + '_'.join(methods),
                   tokenizer=text_enc, max_len=max_len_limit)
    if lp:
        print('_'.join(methods)+' data droped')
    train(sep_flag='\t', sep_num=len(methods),
          train_corpus=data_path + 'train.'+'_'.join(methods),
          dev_corpus=data_path + 'val.'+'_'.join(methods),
          infer_ckpt_path=model_path+'/modern_infer',
          train_ckpt_path=model_path+'/modern_train')
    test(model_path+'/modern_infer', data_path + 'eval.'+'_'.join(methods),
         output_path + 'modern.gpt.'+'_'.join(methods))


def simple_finetune_debug(domain='fr',methods='ori',max_len_limit=220):
    methods=[methods]
    if not os.path.exists('gpt/models/'+domain):
        os.mkdir('gpt/models/'+domain)
    model_path='gpt/models/'+domain+'/'+'_'.join(methods)
    output_path='evaluate/'+domain+'/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
        os.mkdir(model_path+'/neutral_train')
        os.mkdir(model_path+'/neutral_infer')
    data_path = 'training_data/dif_models_'+domain+'/'
    lp = cat_files([data_path + 'biased-test.tok'],
                   data_path + 'test_debug.' + '_'.join(methods),
                   tokenizer=text_enc, max_len=max_len_limit)
    if lp:
        print('_'.join(methods)+' data droped')
    test(model_path+'/neutral_infer', data_path + 'test_debug.'+'_'.join(methods),
         output_path + 'test.neutral.gpt.'+'_'.join(methods))