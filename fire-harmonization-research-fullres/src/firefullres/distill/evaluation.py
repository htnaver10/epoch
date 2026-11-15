"""Matched benchmark comparison, with validation-only frozen operating policies."""
import csv
import json
import sys

import numpy as np
import torch

from .. import core,engine
from ..calibration import sigmoid,fit_intercept,matched_false_alarm_cutoff,metrics


def collect(manifest,checkpoint,split):
    # Reuse the existing MODIS-only inference path, not the distillation loader.
    sys.path.insert(0,str(core.ROOT/'tools'))
    from evaluate_substantial_round import collect as collect_modis
    return collect_modis(manifest,checkpoint,split)


def matching_queries(a,b):
    for key in ('target','weight','modis','tile_index'):
        if not np.array_equal(a[key],b[key]):raise ValueError('Baseline/student evaluation queries differ')


def method_scores(values,offset,cut):
    z,y,w=(values[key] for key in ('logits','target','weight'))
    return {'raw_at_0_5':metrics(sigmoid(z),y,w,z>=0,ranking_score=z),
        'validation_adjusted_frozen_policy':metrics(sigmoid(z.astype(np.float64)+offset),y,w,z.astype(np.float64)>=cut,ranking_score=z)}


def render(report,name):
    path=f'results/{name}'
    lines=['# VIIRS teacher to MODIS student: completed experiment','',
        '**The 2024 test benchmark was examined in the previous experiment. This is a reused benchmark comparison, not a new blind test.**','',
        'The teacher receives calibrated VIIRS I04/I05 thermal imagery and validity masks. The student receives MODIS inputs only.',
        'Teacher soft targets are restricted to training tiles. Both checkpoints are selected using development validation.',
        'New student calibration and its operating threshold are frozen on validation before test prediction.','',
        f'Teacher epochs completed: {report["training"]["teacher_epochs"]}; student epochs completed: {report["training"]["student_epochs"]}.',
        f'Selected teacher epoch: {report["training"]["teacher_selected_epoch"]}; selected student epoch: {report["training"]["student_selected_epoch"]}.','']
    rows=[]
    for split in ('validation','test'):
        lines += ['## '+split.capitalize(),'','| Method | Precision | Recall | TP | FP | FN | Brier | AP |','|---|---:|---:|---:|---:|---:|---:|---:|']
        methods=report[split]
        for key,label in [('modis','Regular MODIS'),('supervised_baseline','Previous MODIS ML, raw 0.5'),
                ('distilled_student','Distilled MODIS student, raw 0.5'),('distilled_student_adjusted','Distilled student, frozen validation policy')]:
            value=methods[key];ap=value['average_precision_exact']
            lines.append(f'| {label} | {100*value["precision"]:.2f}% | {100*value["recall"]:.2f}% | '
                f'{value["true_positives"]:,.0f} | {value["false_positives"]:,.0f} | {value["false_negatives"]:,.0f} | '
                f'{value["brier"]:.8f} | {ap:.4f} |')
            rows.append({'split':split,'method':key,**value})
        lines.append('')
    lines += ['## Interpretation limits','',
        'A student gain must be assessed against both MODIS and the previous supervised model at the stated operating policies.',
        'Changing a threshold after seeing this benchmark would be exploratory. No post-test retuning was performed.',
        'The validation cohort remains limited. Positive query labels within scenes are correlated and are not independent wildfire incidents.',
        'VIIRS provides reference labels, not independently verified ground truth. This experiment does not establish operational usefulness.',
        'The exported student uses the ordinary NativeFireNet loader; teacher models, radiances and soft-target files are unnecessary for inference.','',
        '[Frozen operating policy](operating_policy.json) | [Teacher validation quality](teacher_quality.json) | [Full results](evaluation.json)','']
    core.path(f'{path}/EVALUATION.md',True).write_text('\n'.join(lines),encoding='utf-8')
    with core.path(f'{path}/metrics.csv',True).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def evaluate(config):
    name=config['name'];folder=f'results/{name}';manifest=config['parent_manifest']
    student=f'models/{name}_student/best.pt';teacher=f'models/{name}_teacher/best.pt'
    baseline=config['baseline_checkpoint'];digest=core.sha256(student)
    if core.sha256(manifest)!=config['parent_manifest_sha256'] or core.sha256(baseline)!=config['baseline_checkpoint_sha256']:
        raise ValueError('Frozen parent benchmark or baseline checkpoint changed')
    report_path=f'{folder}/evaluation.json'
    if core.path(report_path).exists():
        existing=core.read_json(report_path)
        if existing['student_checkpoint_sha256']!=digest:raise ValueError('Sealed distillation evaluation belongs to another checkpoint')
        render(existing,name);return existing
    print('Collecting development validation with MODIS-only baseline and student inputs.',flush=True)
    validation,validation_tiles=collect(manifest,student,'validation')
    base_validation,_=collect(manifest,baseline,'validation');matching_queries(validation,base_validation)
    core.write_npz(f'{folder}/validation_student_predictions.npz',**validation)
    policy_path=f'{folder}/operating_policy.json'
    if core.path(policy_path).exists():
        policy=core.read_json(policy_path)
        if policy['student_checkpoint_sha256']!=digest or policy['manifest_sha256']!=core.sha256(manifest):
            raise ValueError('Frozen operating policy belongs to another student or dataset')
    else:
        offset=fit_intercept(validation['logits'],validation['target'],validation['weight'])
        cut,budget=matched_false_alarm_cutoff(validation['logits'],validation['target'],validation['weight'],validation['modis'])
        policy={'student_checkpoint_sha256':digest,'manifest_sha256':core.sha256(manifest),
            'logit_offset':offset,'raw_logit_cutoff':cut,'validation_modis_false_positive_budget':budget,
            'fit_split':'validation','test_used_for_fit':False,'benchmark_previously_examined':True}
        core.write_json(policy_path,policy)
    def scores(student_values,base_values):
        z,y,w,b=(base_values[k] for k in ('logits','target','weight','modis'))
        student_scores=method_scores(student_values,policy['logit_offset'],policy['raw_logit_cutoff'])
        return {'modis':metrics(b,y,w,b>=.5),
            'supervised_baseline':metrics(sigmoid(z),y,w,z>=0,ranking_score=z),
            'distilled_student':student_scores['raw_at_0_5'],
            'distilled_student_adjusted':student_scores['validation_adjusted_frozen_policy']}
    validation_scores=scores(validation,base_validation)
    print('Student operating policy frozen. Evaluating the previously examined 2024 benchmark.',flush=True)
    test,test_tiles=collect(manifest,student,'test');base_test,_=collect(manifest,baseline,'test');matching_queries(test,base_test)
    core.write_npz(f'{folder}/test_student_predictions.npz',**test)
    exported=f'models/{name}_student/student_inference.pt'
    export_model,export=engine.load_checkpoint(exported,torch.device('cpu'))
    if export.get('requires_viirs_at_inference') is not False or tuple(export['inference_input_keys'])!=core.INPUT_KEYS:
        raise ValueError('Student export unexpectedly requires VIIRS inputs')
    selected=torch.load(core.path(student),map_location='cpu',weights_only=True)
    if any(not torch.equal(value,selected['model'][key]) for key,value in export_model.state_dict().items()):
        raise ValueError('Exported student weights differ from the selected checkpoint')
    teacher_saved=torch.load(core.path(teacher),map_location='cpu',weights_only=True)
    teacher_log=core.read_jsonl(f'logs/{name}_teacher/epochs.jsonl');student_log=core.read_jsonl(f'logs/{name}_student/epochs.jsonl')
    report={'scope':config['scope'],'benchmark_previously_examined':True,'new_blind_test':False,
        'manifest_sha256':core.sha256(manifest),'student_checkpoint_sha256':digest,
        'baseline_checkpoint_sha256':core.sha256(baseline),'teacher_checkpoint_sha256':core.sha256(teacher),
        'student_export':exported,'student_export_sha256':core.sha256(exported),'student_export_strict_reload_passed':True,
        'requires_viirs_at_student_inference':False,'operating_policy':policy,
        'training':{'teacher_epochs':len(teacher_log),'student_epochs':len(student_log),
            'teacher_selected_epoch':teacher_saved['epoch']+1,'student_selected_epoch':selected['epoch']+1,
            'temperature':config['temperature'],'distillation_weight':config['distillation_weight']},
        'validation':validation_scores,'test':scores(test,base_test),'validation_tiles':validation_tiles,'test_tiles':test_tiles,
        'model_usefulness_established':False}
    core.write_json(report_path,report);render(report,name)
    print(json.dumps({'test':report['test'],'report':report_path},indent=2),flush=True)
    return report
