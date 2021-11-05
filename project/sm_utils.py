import time

from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from smexperiments.trial_component import TrialComponent
from smexperiments.tracker import Tracker



def create_experiment(project):
    create_date = time.strftime("%Y%m%d")
    exp_name = f"{project}-{create_date}"
    experiment = Experiment.create(experiment_name = exp_name,
                                   description = project,
#                                    tags = [{'Key': 'demo-experiments', 'Value': 'demo1'}]
                                  )
    print(f"{exp_name} experiment created!")
    return experiment


def create_trial(subproject, experiment_name):
    """A trial is a job under an experiment"""
    trial = Trial.create(trial_name = subproject,
                         experiment_name = experiment_name,
#                          sagemaker_boto_client=sm
#                          tags = [{'Key': 'demo-trials', 'Value': 'demo1'}]
                        )
    print(f"{trial_name} trial created!")
    return trial


def logger(trial_component_name, hyperparameters=None, model_history=None):
    """log stuff into trial component
    
    Args:
        hyperparameters (dict): key=parameter_name, value=numeric value
    """
    my_tracker = Tracker.load(trial_component_name=trial_component_name)
    if hyperparameters:
        my_tracker.log_parameter(hyperparameters)


def delete_experiment(experiment_name):
    """delete sagemaker experiments and its components
    https://docs.aws.amazon.com/sagemaker/latest/dg/experiments-cleanup.html"""
    
    experiment = Experiment.load(experiment_name=experiment_name)
    
    for trial_summary in experiment.list_trials():
        trial = Trial.load(trial_name=trial_summary.trial_name)
        for trial_component_summary in trial.list_trial_components():
            tc = TrialComponent.load(trial_component_name=trial_component_summary.trial_component_name)
            trial.remove_trial_component(tc)
            try:
                # comment out to keep trial components
                tc.delete()
            except:
                # tc is associated with another trial
                continue
            # to prevent throttling
            time.sleep(.5)
        trial.delete()
        experiment_name = experiment.experiment_name
    experiment.delete()
    
    print(f"\nExperiment {experiment_name} & its components deleted!")
    
    
    
    
if __name__ == "__main__":
    project="rre"
    subproject="similarity"
    exp = create_experiment(project)
    exp_name = exp.experiment_name
    b = create_trial(subproject, exp_name)
    
    with Tracker.create(display_name="Preprocessing") as tracker:
        tracker.log_parameters({
            "train_test_split_ratio": 0.2,
            "random_state":0
        })
        # we can log the s3 uri to the dataset we just uploaded
        tracker.log_input(name="ccdefault-raw-dataset", media_type="s3/uri", value="wewe\wewe")
        
        print(tracker.trial_component)
    
    b = b.add_trial_component(tracker)
    print(b)
    
#     cleanup_sme_sdk("rre-20211026")
