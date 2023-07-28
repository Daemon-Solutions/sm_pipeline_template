
import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession

from sagemaker.workflow.lambda_step import LambdaStep
from sagemaker.lambda_helper import Lambda

from sagemaker.inputs import CreateModelInput
from sagemaker.workflow.steps import CreateModelStep
from sagemaker.model_metrics import MetricsSource, ModelMetrics 
from sagemaker.workflow.step_collections import RegisterModel


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role="arn:aws:iam::144986566169:role/DaemonMLAISageMakerRole",
    default_bucket=None,
    model_package_group_name="NYCTrafficIncidentsModelPackageGroupName",
    pipeline_name="NYCTrafficIncidentsPipeline",
    base_job_prefix="NYCTraffic",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
):
    """Gets a SageMaker ML Pipeline instance working with NYC traffic collision data

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    print(f"ExecutionRole is {role}")

    pipeline_session = get_pipeline_session(region, default_bucket)

    model_package_group_name = f"NYCTrafficIncidentsModelPackageGroupName"
    input_data_uri = 's3://sagemaker-ml-ops-utility-bucket/sm_pipeline_template/training_data.csv'

    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="Approved"
    )
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=f's3://sagemaker-ml-ops-utility-bucket/sm_pipeline_template/training_data.csv',
    )


    # processing step for feature engineering
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-ny-traffic-preprocess",
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = sklearn_processor.run(
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code="pipelines/nyc_traffic_data/preprocessing.py",
        arguments=["--input-data", input_data],
    )
    
    step_process = ProcessingStep(
        name="TrafficProcess",
        step_args=step_args,
    )

    model_path = f"s3://sagemaker-ml-ops-utility-bucket/sm_pipeline_template/TrafficTrain"

    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type="ml.m5.xlarge"
    )
    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        output_path=model_path,
        role=role,
    )
    xgb_train.set_hyperparameters(
        objective="reg:linear",
        num_round=50,
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.7,
        silent=0
    )

    step_train = TrainingStep(
        name="NYCTrafficIncidentTrain",
        estimator=xgb_train,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv"
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv"
            )
        },
    )


    model = Model(
        image_uri=image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=sagemaker_session,
        role=role,
    )

    inputs = CreateModelInput(
        instance_type="ml.m5.large",
        accelerator_type="ml.eia1.medium",
    )

    step_create_model = CreateModelStep(
        name="NYCTrafficIncidentCreateModel",
        model=model,
        inputs=inputs,
    )


    step_register = RegisterModel(
        name="NYCTrafficIncidentRegisterModel",
        estimator=xgb_train,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status
    )

    step_lambda = LambdaStep(
        name="NotificationLambda",
        lambda_func=Lambda(
            function_arn="arn:aws:lambda:eu-west-2:144986566169:function:SageMakerModelNotifier"
        ),
        inputs={
            "dummy_input": step_register.properties.ModelPackageName,
        },
    )

    pipeline_name = f"NYCTrafficIncidentPipeline"

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_count,
            model_approval_status,
            input_data,
        ],
        steps=[step_process, step_train, step_register, step_create_model, step_lambda],
    )
    return pipeline
