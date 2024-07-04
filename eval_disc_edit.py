import json
import math

tasks = ['whatsup', 'something', 'ag', 'kubric', 'clevr']
ckpts = [
    'checkpoints_magic_reproduce_epoch=000047-step=000012999.ckpt_results.json',
    'logs_logs_finetune_magicbrush_ag_something_kubric_15-15-1-1_init-magic_first_checkpoints_trainstep_checkpoints_step=000041999.ckpt_results.json'
]

for ckpt in ckpts:
    print(ckpt)
    skill_scores_latent_l2 = {task: [] for task in tasks}
    for task in tasks:
        results = json.load(open(f'itm_evaluation/test/{task}/{ckpt}'))
        samples = 4

        for idx, result in results.items():
            pos_latent_l2s = result['pos']['latent_l2']
            neg_latent_l2s = result['neg']['latent_l2']
            if task == 'flickr_edit':
                skills = result['task'].split(',')
                skills = [skill.strip() for skill in skills]
                for skill in skills:
                    skill_scores_latent_l2[skill] += [1 if pos_latent_l2s[i] < neg_latent_l2s[i] else 0 for i in range(len(pos_latent_l2s))]
            skill_scores_latent_l2[task] += [1 if pos_latent_l2s[i] < neg_latent_l2s[i] else 0 for i in range(len(pos_latent_l2s))]

    # make latex row with each task's score
    row = ''
    for k, v in skill_scores_latent_l2.items():
        final_score = sum(v) / len(v)
        se = math.sqrt(final_score * (1 - final_score) / len(v))
        row += f' & {final_score:.3f} \pm {se:.3f}'
    print(row)
