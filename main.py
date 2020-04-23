import os
from gpt.src.concat_fine_tuning import concat_finetuning, domain_combined
from gpt.src.hierarchical_attention import HA
from gpt.src.simple_finetune import simple_finetune, ensemble_test, simple_finetune_debug


def run_simple_finetune_and_emsemble_decoding(domain):
    simple_finetune(domain=domain, methods='ori')
    simple_finetune(domain=domain, methods='rule')
    ensemble_test(domain=domain)


def run_concat_finetune(domain):
    concat_finetuning(domain=domain)

def run_ha(domain):
    HA(domain=domain)

def run_all():
    simple_finetune(domain='biased', methods='ori')
#    simple_finetune_debug(domain='biased', methods='ori')
#    simple_finetune(domain='fr', methods='ori')
#    run_simple_finetune_and_emsemble_decoding('fr')
#    run_simple_finetune_and_emsemble_decoding('em')
#    run_concat_finetune('fr')
#    run_concat_finetune('em')
#    run_ha('fr')
#    run_ha('em')
#    domain_combined('fr',only_test=False)
#    domain_combined('em',only_test=True)

if __name__=='__main__':
    run_all()

